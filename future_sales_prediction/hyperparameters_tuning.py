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


from functools import partial
import logging
import optuna

import mlflow

import time

import click

import pickle

import argparse

import os
import boto3

SOURCE_PATH = None
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
        source_path (str):  path to folder where necessary data stored

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
        source_path (str):  path to folder where necessary data stored
    Yields:
        tuple: 
            - X (pd.DataFrame): Feature batch.
            - Y (pd.Series): Target batch.
    """
    last_monthes_to_take_in_train = 14
    lengthes = np.array(list(map(len, shop_item_pairs_WITH_PREV_in_dbn))) 
    
    total_number_of_samples=sum(lengthes[dbn-last_monthes_to_take_in_train+1:dbn+1])
    number_of_batches = total_number_of_samples // batch_size if batch_size <= total_number_of_samples else 1
    #print(number_of_batches)
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
        dbn (int): Date block number.
        shop_item_pairs_in_dbn (pd.DataFrame): dataframe of cartesian products of (shop, item) for date_block_nums
        batch_size_to_read (int): chunck size when reading csv file. This prevents memory error when using multiple processes
        source_path (str):  path to folder where necessary data stored

    Yields:
        tuple: 
            - X (pd.DataFrame): Feature batch.
            - Y (pd.Series): Target batch.
    """
    val = shop_item_pairs_in_dbn[dbn]
    
    shops = np.unique(list(zip(*val))[0])
    items = np.unique(list(zip(*val))[1])
    
    cartesian_product = np.random.permutation (np.array(np.meshgrid(shops, items)).T.reshape(-1, 2))
    batch_size = len(cartesian_product)+1
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
        l_x = pd.concat(l_x)
        l_y = pd.concat(l_y)


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

def train_model(model, batch_size, val_month, shop_item_pairs_WITH_PREV_in_dbn,batch_size_to_read,batches_for_training,shop_item_pairs_in_dbn,source_path):
    """
    Trains a machine learning model with specified batches and tracks RMSE for training on current val_month.
    Args:
        model (object): Machine learning model to be trained.
        batch_size (int): Number of samples per batch.
        val_month (int): Month used for validation.
        shop_item_pairs_WITH_PREV_in_dbn (np.array[np.array[np.array[int,int]]]):  Accumulated cartesian products for each time block.
        batch_size_to_read (int): chunck size when reading csv file. This prevents memory error when using multiple processes
        batches_for_training (int): number of batches to train on. These parameter may be extremelly usefull for models which doesnt support batch learning. Also this may be usefull for reducing train time
        shop_item_pairs_in_dbn (pd.DataFrame, optional): dataframe of cartesian products of (shop, item) for date_block_nums. If None, validation is not performed after every batch
        source_path (str):  path to folder where necessary data stored

    Returns:
        tuple: 
            - model (object): Trained model.
            - columns_order (list[str]): Ordered list of feature columns.
    """
    first=True
    rmse = 0
    c=0
    columns_order=None
    
    Y_true_l = []
    preds_l = []
    for X_train,Y_train  in create_batch_train(batch_size, val_month,shop_item_pairs_WITH_PREV_in_dbn,batch_size_to_read,source_path):
        #print(f'train on batch {c} started')
        t1_batch = time.time()
        t1_data_prep = time.time()
        #print(f'data preparation on batch {c} started')
        if X_train is None:
            print('None')
            continue

        if type(model) in [Lasso,SVC]:
            #print(X_train.columns)
            X_train.drop('shop_id', inplace=True, axis=1) 
            X_train.drop('item_category_id', inplace=True, axis=1) 
            X_train.drop('item_id', inplace=True, axis=1)
            X_train.drop('city', inplace=True, axis=1)
            X_train.drop('shop_id', inplace=True, axis=1)
        elif type(model) ==LGBMRegressor:
            #print(list(X_train.columns))
        
            X_train = X_train.drop('item_id', axis=1)
            X_train['shop_id'] = X_train['shop_id'].astype('category')
            X_train['item_category_id'] = X_train['item_category_id'].astype('category')
            X_train['city'] = X_train['city'].astype('category')
            X_train['super_category'] = X_train['super_category'].astype('category')
            
            pass
        
        
            
        Y_train = np.clip(Y_train,0,20)
        
        if X_train.empty:
            print('None')
            continue
    
        
        columns_order=X_train.columns

        t2_data_prep = time.time()
        #print(f'data preparation on batch {c} time:',t2_data_prep-t1_data_prep)
        #print('model fitting started')
        t1_fit = time.time()
        if c == 0:
            pass
            #print('train columns')
            #print(X_train.columns)
        if type(model) in [Lasso,SVC]:
            model.fit(X_train, Y_train)
            y_train_pred = model.predict(X_train)
        
        elif type(model) == LGBMRegressor:
            if first:
                model.fit(X_train, Y_train)
                first=False
            else:
                model.fit(X_train, Y_train, init_model=model)
            y_train_pred = model.predict(X_train, validate_features=True)

        

        elif type(model) == RandomForestRegressor:
            model.fit(X_train, Y_train)
            y_train_pred = model.predict(X_train)  
        t2_fit = time.time()
        #print(f'model fitting time on batch {c},',t2_fit - t1_fit)
        
        Y_true_l.append(Y_train)
        preds_l.append(y_train_pred)
        t2_batch = time.time()
        print(f'train on batch {c} time,',t2_batch-t1_batch)
        
        if shop_item_pairs_in_dbn is not None:
            val_pred, val_error = validate_model(model, val_month,columns_order, shop_item_pairs_in_dbn,batch_size_to_read,source_path)
            print(f'val score after batch {c}', val_error)

        c+=1

        if c == batches_for_training:
            break
        
    train_rmse = root_mean_squared_error(pd.concat(Y_true_l), np.concat(preds_l))
    print('train_rmse, ',train_rmse)
           

    return model, columns_order

def validate_model(model, val_month, columns_order, shop_item_pairs_in_dbn, batch_size_to_read,source_path):
    """
    Validates the model and calculates RMSE on the validation set on the current val_month.

    Args:
        model (object): Machine learning model to be validated.=
        val_month (int): Month used for validation.
        columns_order (list[str]): Order of feature columns.
        shop_item_pairs_in_dbn (pd.DataFrame): dataframe of cartesian products of (shop, item) for date_block_nums
        batch_size_to_read (int): chunck size when reading csv file. This prevents memory error when using multiple processes
        source_path (str):  path to folder where necessary data stored

    Returns:
        tuple: 
            - val_preds (np.array[float]): Predictions for validation set.
            - val_rmse (float): RMSE for validation set.
    """
    rmse = 0
    c=0
    
    val_preds = []
    Y_true_l = []
    preds_l = []
    
    for X_val, Y_val in create_batch_val(val_month, shop_item_pairs_in_dbn, batch_size_to_read,source_path):#but then cartesian product used
        if X_val is None:
                    continue
        
        if type(model) in [sklearn.linear_model._coordinate_descent.Lasso,
                          SVC]:
            
            X_val.drop('shop_id', inplace=True, axis=1) 
            X_val.drop('item_category_id', inplace=True, axis=1) 
            X_val.drop('item_id', inplace=True, axis=1)
            X_val.drop('city', inplace=True, axis=1)
            X_val.drop('shop_id', inplace=True, axis=1)
            

        elif type(model) ==LGBMRegressor:
            
            X_val = X_val.drop('item_id', axis=1)
            X_val['shop_id'] = X_val['shop_id'].astype('category')
            X_val['item_category_id'] = X_val['item_category_id'].astype('category')
            X_val['city'] = X_val['city'].astype('category')
            X_val['super_category'] = X_val['super_category'].astype('category')
                    
            pass
            
        
            
        Y_val = np.clip(Y_val,0,20)
        
        
        X_val = make_X_lag_format(X_val, val_month)
        
        X_val=append_some_columns(X_val, val_month)
        X_val = X_val[columns_order]

        if type(model) in [Lasso,SVC]:
            y_val_pred = model.predict(X_val)#lgb - validate features
            
        elif type(model) ==LGBMRegressor:
            y_val_pred = model.predict(X_val, validate_features=True)#lgb - validate features

           

        elif type(model) == RandomForestRegressor:
            y_val_pred = model.predict(X_val)  

        y_val_pred = np.clip(y_val_pred,0,20)

        
        
        
        preds_l.append(y_val_pred)
        Y_true_l.append(Y_val)
        
        c+=1
        
        val_preds.append(y_val_pred)

    
    val_rmse = root_mean_squared_error(pd.concat(Y_true_l), np.concat(preds_l))
    print('val rmse, ',val_rmse)

    return val_preds, val_rmse

def validate_ML(params,batch_size,val_monthes, shop_item_pairs_in_dbn, shop_item_pairs_WITH_PREV_in_dbn,batch_size_to_read, batches_for_training,source_path,experiment_id):
    """
    Runs model training and validation across multiple months and computes RMSE for each month.
    Note: batch learning works properly only for LGBMRegressor. Using another model with batch_size<=len(shop_item_pairs_WITH_PREV_in_dbn[dbn]) will lead to incorrect results
    Args:
        params (dict): parameters for machine learning model.
        batch_size (int): Number of samples per batch.
        val_monthes (list): List of months for validation.
        shop_item_pairs_in_dbn (pd.DataFrame): dataframe of cartesian products of (shop, item) for date_block_nums
        shop_item_pairs_WITH_PREV_in_dbn ( np.array[np.array[np.array[int,int]]] ): array of accumulated cartesian products of (shop, item) for date_block_nums
        batch_size_to_read (int): chunck size when reading csv file. This prevents memory error when using multiple processes
        batches_for_training (int): number of batches to train on. These parameter may be extremelly usefull for models which doesnt support batch learning. Also this may be usefull for reducing train time
        source_path(str): path to folder where necessary data stored
        experiment_id (str): mlflow experiment id
    Returns:
        tuple: 
            - val_errors (list[np.array[float]]): List of RMSE values for each month.
            - val_preds (list[float]): Predictions for validation set.
    """
    
    val_errors = []
    
    val_preds=[]
    
    
    for val_month in val_monthes:
        
        run_name=f'validation on {val_month}'
        with mlflow.start_run(experiment_id=experiment_id, run_name=run_name,nested=True):

            model = LGBMRegressor(verbose=-1,n_jobs=multiprocessing.cpu_count(), num_leaves=params['num_leaves'],
                                n_estimators = params['n_estimators'],
                                    learning_rate=params['lr'])
            
            
            print(f'month {val_month} started')
            t1 = time.time()
            
            print('month', val_month%12)
            model,columns_order = train_model(
                model, 
                batch_size, 
                val_month,
                shop_item_pairs_WITH_PREV_in_dbn,
                batch_size_to_read,
                batches_for_training,
                shop_item_pairs_in_dbn = None,
                source_path=source_path
            )

            t2=time.time()
            print(f'model training on {val_month} time,',t2-t1)
            print('feature importances, ')
            print(list(model.feature_names_in_[np.argsort( model.feature_importances_)][::-1]))
            
            t1 = time.time()

            val_pred, val_error = validate_model(
                model,
                val_month,
                columns_order,
                shop_item_pairs_in_dbn,
                batch_size_to_read,
                source_path
            )

            t2 = time.time()
            print(f'validation time on month {val_month},',t2-t1)
            val_errors.append(val_error)
            val_preds.append(val_pred)

            mlflow.log_metric('rmse',val_error )
            mlflow.log_params(params)

            
            
            
    return val_errors, val_preds


    


def objective(trial,val_monthes,shop_item_pairs_in_dbn,shop_item_pairs_WITH_PREV_in_dbn,source_path,experiment_id,
              max_num_leaves_range_optuna,batch_size_for_train,lr_range_optuna,num_estimators_range_optuna ):
    
    
    with mlflow.start_run(experiment_id=experiment_id, run_name='run opt',nested=True):
        val_monthes_str = [str(i) for i in val_monthes]
        lr = trial.suggest_float('lr', low=lr_range_optuna[0], high = lr_range_optuna[1],log=True)
        num_leaves = trial.suggest_int('num_leaves', low=max_num_leaves_range_optuna[0], high = max_num_leaves_range_optuna[1],step=50)
        n_estimators=trial.suggest_int('n_estimators', low=num_estimators_range_optuna[0], high = num_estimators_range_optuna[1],step=50)

        #parametrs which doesnt affect training
        batch_size_to_read=50_000 # the more batches_for_training - the less should be batch_size_to_read to prevent memory error

        #parametrs which affect number of estimators:
        batches_for_training=1#each batch increases number of estimators with n_estimators
        batch_size=batch_size_for_train #(real batch size will be a bit different from this). In fact this is used only to find number of batchs


        print('validation started...')
        params = defaultdict()
        params['trial'] = trial
        params['lr'] = lr
        params['num_leaves'] = num_leaves
        params['n_estimators'] = n_estimators
        params['batch_size']=batch_size
        params['batches_for_training']=batches_for_training
        params['batch_size_to_read']=batch_size_to_read
        val_errors, val_preds = validate_ML(
                                            params,
                                            batch_size=batch_size,
                                            val_monthes=val_monthes, 
                                            shop_item_pairs_in_dbn=shop_item_pairs_in_dbn,
                                            shop_item_pairs_WITH_PREV_in_dbn=shop_item_pairs_WITH_PREV_in_dbn,
                                            batch_size_to_read=batch_size_to_read,
                                            batches_for_training=batches_for_training,
                                            source_path=source_path,
                                            experiment_id=experiment_id
                                            )

        print('lr:'+str(lr) + ' ' + str(val_errors) + '\n')

        mlflow.log_params(params)
        mlflow.log_metric('rmse', np.mean(val_errors))
        return np.mean(val_errors)

def run_optimizing(args, experiment_id, test_month):
    """
    Function for hyperparameters tuning. Writes best parametrs to ./saved_dictionary.pkl

    Args:
        args (): parsed arguments from CLI
        experiment_id (str): experiment id for mlflow
        test_month (int): 
    """
    path_for_merged = args.path_for_merged
    path_data_cleaned= args.path_data_cleaned
    n_trials=args.n_trials

    max_num_leaves_range_optuna = args.max_num_leaves_range_optuna
    batch_size_for_train = args.batch_size_for_train
    lr_range_optuna=args.lr_range_optuna
    num_estimators_range_optuna = args.num_estimators_range_optuna

    print(batch_size_for_train)


    data_train = pd.read_csv(f'{path_data_cleaned}/data_train.csv')
    test = pd.read_csv(f'{path_data_cleaned}/test.csv')
    test['date_block_num'] = test_month
    data_train = pd.concat([data_train,test ], ignore_index=True).drop('ID', axis=1).fillna(0)


    shop_item_pairs_in_dbn, shop_item_pairs_WITH_PREV_in_dbn = prepare_past_ID_s_CARTESIAN(data_train)

    val_monthes=[test_month-12,test_month-9,test_month-3,test_month-1]

    val_monthes_str = [str(i) for i in val_monthes]

    objective_f = partial(objective,
                         val_monthes = val_monthes,
                         shop_item_pairs_in_dbn=shop_item_pairs_in_dbn,
                         shop_item_pairs_WITH_PREV_in_dbn=shop_item_pairs_WITH_PREV_in_dbn,
                         source_path=path_for_merged,
                         experiment_id=experiment_id,

                         max_num_leaves_range_optuna=max_num_leaves_range_optuna,
                         batch_size_for_train=batch_size_for_train,
                         lr_range_optuna=lr_range_optuna,
                         num_estimators_range_optuna=num_estimators_range_optuna)

    study = optuna.create_study(direction="minimize")

    study.optimize(objective_f, n_trials=n_trials)

    with open('best_parameters.pkl', 'wb') as f:
        pickle.dump(study.best_params, f)

    return study.best_params


def download_s3_folder(s3c, bucket_name, s3_folder, local_dir=None):
    """
    Download the contents of a folder directory to local_dir (creates if not exist)
    Args:
        bucket_name: the name of the s3 bucket
        s3_folder: the folder path in the s3 bucket
        local_dir: a relative or absolute directory path in the local file system
    """
    bucket = s3c.Bucket(bucket_name)
    for obj in bucket.objects.filter(Prefix=s3_folder):
        target = obj.key if local_dir is None \
            else os.path.join(local_dir, os.path.relpath(obj.key, s3_folder))
        if not os.path.exists(os.path.dirname(target)):
            os.makedirs(os.path.dirname(target))
        if obj.key[-1] == '/':
            continue
        bucket.download_file(obj.key, target)
        #print(obj.key, target)


def get_or_create_experiment(experiment_name):
    """
    Retrieve the ID of an existing MLflow experiment or create a new one if it doesn't exist.

    This function checks if an experiment with the given name exists within MLflow.
    If it does, the function returns its ID. If not, it creates a new experiment
    with the provided name and returns its ID.

    Parameters:
    - experiment_name (str): Name of the MLflow experiment.

    Returns:
    - str: ID of the existing or newly created MLflow experiment.
    """

    if experiment := mlflow.get_experiment_by_name(experiment_name):
        return experiment.experiment_id
    else:
        return mlflow.create_experiment(experiment_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--test_month', type=int, help='month to create submission on')

    parser.add_argument('--run_name', type=str)
    parser.add_argument('--path_for_merged', type=str, help='folder where merged.csv stored after prepare_data.py. Also best hyperparams will be stored here')
    parser.add_argument('--path_data_cleaned', type=str, help='folder where data stored after etl')
    parser.add_argument('--n_trials', type=int, help='Number of iteration for TPE estimator')

    parser.add_argument('--max_num_leaves_range_optuna',nargs=2, type=int)
    parser.add_argument('--lr_range_optuna',nargs=2, type=float)
    parser.add_argument('--num_estimators_range_optuna',nargs=2, type=int)
    parser.add_argument('--batch_size_for_train', type=int)

    args = parser.parse_args()

    minio_user=os.environ.get("MINIO_ACCESS_KEY")
    minio_password=os.environ.get("MINIO_SECRET_ACCESS_KEY")
    bucket_name = os.environ.get("BUCKET_NAME")

    #print(args.batch_size_for_train)
    ACCESS_KEY = minio_user
    SECRET_KEY = minio_password
    host = 'http://minio:9000'
    bucket_name = bucket_name

    s3c = boto3.resource('s3', 
                    aws_access_key_id=ACCESS_KEY,
                    aws_secret_access_key=SECRET_KEY,
                    endpoint_url=host) 
    
    download_s3_folder(s3c,bucket_name,args.path_data_cleaned, args.path_data_cleaned)
    

    s3c = boto3.client('s3', 
                    aws_access_key_id=ACCESS_KEY,
                    aws_secret_access_key=SECRET_KEY,
                    endpoint_url=host)
    
    s3c.download_file(bucket_name, f'{args.path_for_merged}/merged.csv', f'{args.path_for_merged}/merged.csv') # ASSUMES THAT path args.path_for_merged exists

    mlflow.set_tracking_uri(uri="http://mlflow:5000")
    exp = get_or_create_experiment('hyperparameter_optimiztion')

    with mlflow.start_run(experiment_id=exp, run_name=args.run_name):
        
        bp =run_optimizing(args,exp,args.test_month)

    s3c.upload_file('best_parameters.pkl', bucket_name, f'{args.run_name}/best_parameters.pkl')


    s3c.close()