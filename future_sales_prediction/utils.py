import pandas as pd
import numpy as np
import sklearn
from sklearn.metrics import root_mean_squared_error
from collections import defaultdict


from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression

import multiprocessing

import time

import pickle


import boto3
import os

import argparse

import mlflow
#np.random.seed(42)

def prepare_past_ID_s(data_train):
    """
    This function doesn't used
    """
    data_train['shop_item'] = [tuple([shop, item]) for shop, item in zip(data_train['shop_id'], data_train['item_id'])]
    #34 block contains A LOT more shop_item than others
    shop_item_pairs_in_dbn = data_train.groupby('date_block_num')['shop_item'].apply(np.unique)
    data_train = data_train.drop(['shop_item'], axis=1)
    
    shop_item_pairs_WITH_PREV_in_dbn = shop_item_pairs_in_dbn.copy()
    
    print(np.array(shop_item_pairs_WITH_PREV_in_dbn.index))
    arr = np.array(shop_item_pairs_WITH_PREV_in_dbn.index)
    
    for block in arr[arr>=0]:
        if block == 0:
            continue

        
        arr = np.append(shop_item_pairs_WITH_PREV_in_dbn[block -1],
                                                            shop_item_pairs_in_dbn[block-1])
        
        
        shop_item_pairs_WITH_PREV_in_dbn[block] = np.unique(np.append(shop_item_pairs_WITH_PREV_in_dbn[block -1],
                                                            shop_item_pairs_in_dbn[block-1]))
        print(len(shop_item_pairs_WITH_PREV_in_dbn[block]))

   
    return shop_item_pairs_in_dbn, shop_item_pairs_WITH_PREV_in_dbn

def prepare_past_ID_s_CARTESIAN(data_train):
    """
    Prepares unique (shop, item) pairs in a Cartesian product format over time blocks.
    
    Args:
        data_train (pd.DataFrame): Training data with 'shop_id', 'item_id', and 'date_block_num' columns.

    Returns:
        tuple: A tuple containing:
            - shop_item_pairs_in_dbn (pd.DataFrame): Cartesian product of shop_id and item_id columns from data_train for each 'date_block_num'.
            - shop_item_pairs_WITH_PREV_in_dbn:np.array[np.array[np.array[int]]] Accumulated cartesian products for each time block up since 0 to the previous block. This name may confuse, as it contains only previous information
            
    """
    data_train['shop_item'] = [tuple([shop, item]) for shop, item in zip(data_train['shop_id'], data_train['item_id'])]
    #34 block contains A LOT more shop_item than others
    shop_item_pairs_in_dbn = data_train.groupby('date_block_num')['shop_item'].apply(np.unique)
    data_train = data_train.drop(['shop_item'], axis=1)
    
    shop_item_pairs_WITH_PREV_in_dbn = np.array([None] * len(shop_item_pairs_in_dbn))
    
    #print(np.array(shop_item_pairs_WITH_PREV_in_dbn.index))
    

    cartesians = []
    for dbn in shop_item_pairs_in_dbn.index:
        val = shop_item_pairs_in_dbn[dbn]

        shops = np.unique(list(zip(*val))[0])
        items = np.unique(list(zip(*val))[1])
    
        cartesian_product = np.random.permutation (np.array(np.meshgrid(shops, items)).T.reshape(-1, 2))
        #print(cartesian_product)
        cartesians.append(cartesian_product)
        
    
    shop_item_pairs_WITH_PREV_in_dbn[0] = cartesians[0]
    
    for block in shop_item_pairs_in_dbn.index:
        if block == 0:
            continue
        arr = np.append(shop_item_pairs_WITH_PREV_in_dbn[block - 1],
                             cartesians[block - 1], axis=0)#shop_item_pairs_WITH_PREV_in_dbn doesnt contain 34 month
        
        shop_item_pairs_WITH_PREV_in_dbn[block] = np.unique(arr, axis=0)
        print(len(shop_item_pairs_WITH_PREV_in_dbn[block]))

    for i in range(len(shop_item_pairs_WITH_PREV_in_dbn)):
        shop_item_pairs_WITH_PREV_in_dbn[i] = np.random.permutation(shop_item_pairs_WITH_PREV_in_dbn[i])
    return shop_item_pairs_in_dbn, shop_item_pairs_WITH_PREV_in_dbn

def make_X_lag_format(data, dbn):
    """
    Converts columns with date block numbers to a lag format for specified date block number.

    Args:
        data (pd.DataFrame): Data containing columns with date block numbers. Columns must have format name$dbn
        dbn (int): Current date block number for calculating lags.

    Returns:
        pd.DataFrame: Data with lagged columns for the specified date block number. Columns will have format name_lag;{lag number}
    """
    
    lag_cols = defaultdict()
    for col in data.columns:
        splitted = col.split('$')
        if len(splitted) == 1:
            continue
        
        lag_cols[col] = splitted[0] + '_lag;' + str(dbn - int(splitted[1]))

    #print(lag_cols)
    data = data.rename(columns=dict(lag_cols))
    #print(data.columns)
    return data

def prepare_train(data, valid ):
    """
    Filters training data to include only the specified shop-item pairs.

    Args:
        data (pd.DataFrame): Training data to be filtered.
        valid np.array[np.array[int]]: shop, item pairs to include in a batch.

    Returns:
        pd.DataFrame: Filtered data with only specified shop-item pairs.
    """
    #print(data)
    valid_shop_item = valid
    valid_shop_item = list(zip(*valid_shop_item))
    df = pd.DataFrame({'item_id':valid_shop_item[1],'shop_id':valid_shop_item[0]} )
    data = df.merge(data, on=['shop_id','item_id'], how='inner').fillna(0)

    return data


def prepare_val(data, valid ):
    """
    Filters validation data to include only specified shop-item pairs.

    Args:
        data (pd.DataFrame): Validation data to be filtered.
        valid (np.array[np.array[int]]): shop, item pairs to include in a batch.
    Returns:
        pd.DataFrame: Filtered data for validation with only specified shop-item pairs.
    """
    
    df = pd.DataFrame({'item_id':valid[:,1],'shop_id':valid[:,0]} )
    data = df.merge(data, on=['shop_id','item_id'], how='inner').fillna(0)
    #print('prepare_val, data:',len(data))
    return data

def prepare_data_train_boosting(data, valid, dbn):
    """
    Prepares validation data for boosting models by selecting required columns and selecting only required (shop,item) pairs
    This function was used before reading only part of columns from csv, but still be used to validate everything is right.

    Args:
        data (pd.DataFrame): Training data.
        valid (np.array[np.array[int]]): shop, item pairs to include in a batch.
        dbn (int): Current date block number.

    Returns:
        tuple: 
            - X (pd.DataFrame): Features for training.
            - Y (pd.Series): Target variable for training.
    """
    train = prepare_train (data, valid)
    lag_cols = []
    for col in data.columns:
        
        splitted = col.split('$')
        if len(splitted) == 1:
                lag_cols.append(col)
                continue
        #if 'shop_item_cnt' not in col:
        #    continue
            
        for db in range(0,dbn-1):
            
            if db == int(splitted[1]):
                #print(col)
                lag_cols.append(col)

    #print(lag_cols)
    X = train[lag_cols]
    Y = train[f'value_shop_id_item_id${dbn-1}']
    
    return X, Y


def prepare_data_validation_boosting(data, valid, dbn):
    """
    Prepares validation data for boosting models by selecting required columns and selecting only required (shop,item) pairs
    This function was used before reading only part of columns from csv, but still be used to validate everything is right.
    Args:
        data (pd.DataFrame): Validation data.
        valid (np.array[np.array[int]]): shop, item pairs to include in a batch.
        dbn (int): Current date block number.

    Returns:
        tuple: 
            - X (pd.DataFrame): Features for validation.
            - Y (pd.Series): Target variable for validation.
    """
    test = prepare_val (data, valid)
    
    lag_cols = []
    for col in test.columns:
        
            
        splitted = col.split('$')
        if len(splitted) == 1:
                lag_cols.append(col)
                continue
        for db in range(1,dbn):
            
            if db == int(splitted[1]):
                #print(db, int(''.join(re.findall(r'\d+', col))))
                lag_cols.append(col)

    X = test[lag_cols]
    Y = test[f'value_shop_id_item_id${dbn}']
    
    return X, Y

def select_columns_for_reading(path, dbn):
    """
    Selects relevant columns to read from a CSV file based on the date block number. Feature selection step done here

    Args:
        path (str): Path to the CSV file.
        dbn (int): Date block number used to filter columns.

    Returns:
        list: List of column names to read.
    """
    columns = pd.read_csv(path, nrows=0).columns.tolist()

    cols = []
    for col in columns:
        l = col.split('$')
        if len(l) == 1:
            cols.append(col)
            continue

        name = l[0]
        num=int(l[1])
        dbn_diff = dbn - num
        
        if 'value_shop_id_item_id' in col and np.isclose(dbn_diff,0):#target
            cols.append(col)

        if dbn_diff<=0:
            continue


        if 'ema' in name and dbn_diff <= 2:
            cols.append(col)
            continue
        elif 'value_shop_id_item_id' in name and (dbn_diff <= 3 or dbn_diff == 6 or dbn_diff == 12):
            cols.append(col)
            continue
        elif 'value_price' in name and dbn_diff <= 1:
            cols.append(col)
            continue
        elif 'value' in name and dbn_diff <= 3:
            cols.append(col)
            continue
        elif 'diff' in name and dbn_diff == 1:
            cols.append(col)
            continue
        elif 'change' in name and dbn_diff <= 2:
            cols.append(col)
            continue


    return cols

def sample_indexes(shop_item_pairs_WITH_PREV_in_dbn, number_of_batches):
    """
    Samples indexes for batch learning. Indexes must be applied on shop_item_pairs_WITH_PREV_in_dbn
    Args:
        shop_item_pairs_WITH_PREV_in_dbn ( np.array[np.array[np.array[int]]] ): array of accumulated cartesian products of (shop, item) for date_block_nums
        batch_size (int): batch size for lgbm
        number_of_batches (int): number of batches for lgbm,

    Returns:
        list[list[tuple]]: list of batch indexes for each date_block_num. 
    """
    
    lengthes = np.array(list(map(len, shop_item_pairs_WITH_PREV_in_dbn)))

    batch_size_in_dbn =lengthes // number_of_batches
    #print(lengthes)

    to_ret=[]#this will contain last_monthes_to_take_in_train array, where each array contains split for batches for some dbn
    for bs in batch_size_in_dbn:
        idxs_in_dbn = []
        for i in range(number_of_batches):
            idxs_in_dbn.append((i*bs,(i+1)*bs))
        #print(np.max(idxs_in_dbn))
        to_ret.append(idxs_in_dbn)

    return to_ret
def prepare_batch_per_dbn(dbn, dbn_inner,last_monthes_to_take_in_train, shop_item_pairs_WITH_PREV_in_dbn,idxs,batch_size_to_read,source_path):
    """
    Function to execute on a single process (creates part of batch for one date block num)
    Args:
        dbn (int): current date block num
        dbn_inner (int): index of dbn in inner loop in range(last_monthes_to_take_in_train)
        last_monthes_to_take_in_train (_type_): _description_
        shop_item_pairs_WITH_PREV_in_dbn (_type_): _description_
        idxs (): indexes for shop_item_pairs_WITH_PREV_in_dbn for current date block num and batch
        batch_size_to_read (int): chunck size when reading csv file. This prevents memory error when using multiple processes

    Returns:
        list: list of X,Y for current date_block_num
    """
    curr_dbn = ((dbn_inner+1) - last_monthes_to_take_in_train) + dbn # in [dbn-last_monthes_to_take_in_train+1,dbn]
    #idxs[dbn_inner] contains batch indexes for dbn_inner block(this array contains elements for [dbn-last_monthes_to_take_in_train+1,dbn] date_block_nums
    
    train = shop_item_pairs_WITH_PREV_in_dbn[idxs[0] : idxs[1] ]


    columns = select_columns_for_reading(f'{source_path}/merged.csv',curr_dbn-1)
    #(dbn-1) because this dbn is for validation, dbn for train is 1 less

    merged = pd.read_csv(f'{source_path}/merged.csv', chunksize=batch_size_to_read, skipinitialspace=True, usecols=columns)
    l_sum = 0
    l_x=[]
    l_y=[]
    for chunck in merged:#split merged into chuncks
        
        l =  prepare_data_train_boosting(chunck,train, curr_dbn) 


        l_0 = make_X_lag_format(l[0], curr_dbn-1)#-1 because this dbn is for validation, dbn for train is 1 less

        l_0=append_some_columns(l_0, curr_dbn-1)
        
        l_sum += len(l[0])
        l_x.append( l_0 )
        l_y. append(l[1])
    

    l_x = pd.concat(l_x, copy=False)
    l_y = pd.concat(l_y, copy=False)
    return [l_x, l_y]


def create_batch_train(batch_size, dbn, shop_item_pairs_WITH_PREV_in_dbn, batch_size_to_read,source_path):
    """
    Creates training batches for date_block_num.

    Args:
        batch_size (int): Number of samples per batch.
        dbn (int): Date block number.
        shop_item_pairs_WITH_PREV_in_dbn (np.array[np.array[np.array[int]]]):  Accumulated cartesian products for each time block.
        batch_size_to_read (int): chunck size when reading csv file. This prevents memory error when using multiple processes

    Yields:
        tuple: 
            - X (pd.DataFrame): Feature batch.
            - Y (pd.Series): Target batch.
    """
    last_monthes_to_take_in_train = 14
    lengthes = np.array(list(map(len, shop_item_pairs_WITH_PREV_in_dbn))) 
    
    total_number_of_samples=sum(lengthes[dbn-last_monthes_to_take_in_train+1:dbn+1])
    number_of_batches = total_number_of_samples // batch_size if batch_size <= total_number_of_samples else 1

    #This variable will store batch indexes for date_block_nums [ dbn-last_monthes_to_take_in_train+1 , dbn]
    idxs = sample_indexes(shop_item_pairs_WITH_PREV_in_dbn[dbn-last_monthes_to_take_in_train+1:dbn+1],number_of_batches)
    print('total batches,',number_of_batches)
    for batch_number in range(number_of_batches):
        
        l_x=[]
        l_y=[]
        t1 = time.time()
        #This loop enables to includa data from different monthes in one batch
        with multiprocessing.Pool( processes=multiprocessing.cpu_count() ) as pool:
            result  = pool.starmap(prepare_batch_per_dbn, \
                                    [[dbn, 
                                      dbn_inner,
                                      last_monthes_to_take_in_train, 
                                      shop_item_pairs_WITH_PREV_in_dbn[((dbn_inner+1) - last_monthes_to_take_in_train) + dbn ],
                                      idxs[dbn_inner][batch_number],
                                      batch_size_to_read,
                                      source_path] for dbn_inner in range(last_monthes_to_take_in_train)])

        #print(result)
        l_x = [result[i][0] for i in range(len(result))]
        l_y = [result[i][1] for i in range(len(result))]
        #print(l_x)
        t2 = time.time()
        print('batch creation time [create_batch_train, 212],',t2-t1)
                
        
        l_x = pd.concat(l_x, copy=False)
        l_y = pd.concat(l_y, copy=False)

        l_x.reset_index(drop=True, inplace=True)
        l_y.reset_index(drop=True, inplace=True)

        index_perm = np.random.permutation(l_x.index)[:batch_size]

        l_x=l_x.iloc[index_perm]
        l_y = l_y.iloc[index_perm]

        print('batch size', len(l_x))
        print(f'batch {batch_number} memory usage',np.sum(l_x.memory_usage()) / 10**6)
        yield [l_x, l_y]#, test


def create_batch_val(dbn, shop_item_pairs_in_dbn, batch_size_to_read,source_path):
    """
    Creates validation batches for date_block_num.

    Args:
        batch_size (int): Number of samples per batch.
        dbn (int): Date block number.
        shop_item_pairs_in_dbn (pd.DataFrame): dataframe of cartesian products of (shop, item) for date_block_nums
        batch_size_to_read (int): chunck size when reading csv file. This prevents memory error when using multiple processes

    Yields:
        tuple: 
            - X (pd.DataFrame): Feature batch.
            - Y (pd.Series): Target batch.
    """
    val = shop_item_pairs_in_dbn[dbn]
    
    shops = np.unique(list(zip(*val))[0])
    items = np.unique(list(zip(*val))[1])

    cartesian_product = np.random.permutation (np.array(np.meshgrid(shops, items)).T.reshape(-1, 2))

    batch_size= len(cartesian_product)+1
    
    chunk_num =  len(cartesian_product)// batch_size if len(cartesian_product)%batch_size==0  else   len(cartesian_product) // batch_size + 1#MAY BE NEED TO CORRECT


    columns = select_columns_for_reading(f'{source_path}/merged.csv', dbn)

    
    for idx in range(chunk_num):
        merged = pd.read_csv(f'{source_path}/merged.csv', chunksize=batch_size_to_read, skipinitialspace=True, usecols=columns)
        l_x=[]
        l_y=[]
        l_sum=0
        cartesian = cartesian_product[idx*batch_size:(idx+1)*batch_size]

        for chunck in merged:

            l =  prepare_data_validation_boosting(chunck,cartesian, dbn) 
            l_sum+=len(l[0])
            l_x.append( l[0] )
            l_y. append( l[1] )

        if len(l_x) == 0:
            yield [None, None]
        print('create_batch_val,243:',l_sum)
        l_x = pd.concat(l_x, copy=False)
        l_y = pd.concat(l_y, copy=False)


        yield [l_x,l_y]#, test


def append_some_columns(X_train, dbn):
    """
    Adds additional columns like date_block_num and month to the training data.

    Args:
        X_train (pd.DataFrame): Training data.
        dbn (int): Current date block number.

    Returns:
        pd.DataFrame: Training data with additional columns.
    """
    X_train['date_block_num'] = dbn
    X_train['month'] = dbn%12
    return X_train
