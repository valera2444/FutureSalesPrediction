import numpy as np
import pandas as pd

import gc
import time

FIRST_N_MONTH_TO_DROP = 6

def prepare_past_ID_s_CARTESIAN(data_train):
    """
    Prepares cartesian product of unique shop, item pairs over time blocks.
    
    Args:
        data_train (pd.DataFrame): Training data with 'shop_id', 'item_id', 'date_block_num' and some more columns.

    Returns:
        tuple: A tuple containing:
            - shop_item_pairs_in_dbn (pd.DataFrame): Cartesian product of shop_id and item_id columns from data_train for each 'date_block_num'.
            - shop_item_pairs_WITH_PREV_in_dbn (np.array[np.array[np.array[int]]]): Accumulated cartesian products for each time block up since 0 to the current block.
    """
    data_train['shop_item'] = [tuple([shop, item]) for shop, item in zip(data_train['shop_id'], data_train['item_id'])]
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
                             cartesians[block], axis=0)
        
        shop_item_pairs_WITH_PREV_in_dbn[block] = np.unique(arr, axis=0)
        #print(len(shop_item_pairs_WITH_PREV_in_dbn[block]))
    return shop_item_pairs_in_dbn, shop_item_pairs_WITH_PREV_in_dbn

def create_change(table):
    """
    Calculates percentage change for each consecutive month column in the table.
    
    Args:
        table (pd.DataFrame): Table with numerical month columns representing metrics over time.

    Returns:
        pd.DataFrame: Table with calculated percentage changes for each month column.
    """
    table=table.copy().fillna(0)

    cols = [col for col in table.columns if col.isdigit()]
    #print(cols)
    cols.sort()
    result = table.copy()
    for col in cols[1:]:
    
        result[col] = (table[col] / table[str(int(col)-1)] - 1) * 100
        
    result = result.replace([np.inf], 0)
    result = result.replace([np.nan], 0)
    return result
        

import time

def create_pivot_table(data,index, item_shop_city_category_sup_category,column='item_cnt_day'):
    """
    Creates a pivot table with column aggregated by specified index.
    
    Args:
        data (pd.DataFrame): Data with relevant columns and date blocks.
        index (list of str): List of columns to group data by in the pivot table.
        item_shop_city_category_sup_category (pd.DataFrame): Datarame with unique [shop, item] pairs. Used to return data with all requierd [shop, item] pairs.
        column (str): columns to aggregate; default is 'item_cnt_day'.

    Returns:
        pd.DataFrame: Merged pivot table with specified index columns.
    """
    if column == 'item_cnt_day':
        agg = 'mean'
    else:
        agg='mean'
    df = pd.pivot_table(data, 
                    index=index,
                    columns=['date_block_num'],
                    values=[column],
                    aggfunc=agg, 
                    fill_value=0.0
                   ).reset_index()
    
    train = pd.DataFrame({col:df[col] for col in index})
    
    item_cnt_day = df[column]
    train = pd.concat([train,item_cnt_day], axis=1)
    #print(items)
    merged = item_shop_city_category_sup_category.merge(train, how='left').fillna(0)#shop_item which not in data but in cartesian will be 0

    merged.columns = [str(col) for col in merged.columns]
    return merged

def calculate_ema_6(table, alpha=2/(6+1)):
    """
    Calculates the 6-month Exponential Moving Average (EMA) for each record and month.
    
    Args:
        table (pd.DataFrame): Data with columns for each month containing metrics.
        alpha (float): EMA smoothing factor, default set for a 6-month EMA.

    Returns:
        pd.DataFrame: Table with EMA values calculated for each month.
    """

    EMAs = pd.DataFrame(columns = table.columns)
    EMAs['shop_id'] = table['shop_id']
    EMAs['item_id'] = table['item_id']
    EMA_period = 6
    weights = (1 - alpha) ** np.arange(6)[::-1]#???
    print('EMAs weights:',weights)
    
    for dbn in range(EMA_period, 35):#???(35)
        lag_cols =  [str(i) for i in range(dbn - EMA_period+1, dbn+1)]
        
        df_in = table[lag_cols].fillna(0)
        
        weighted_sums = np.dot(df_in[lag_cols].values, weights)
        
        EMAs[str(dbn)] = weighted_sums / sum(weights)
    
    return EMAs



def calculate_EMAs_pipeline(source, groupby, alpha=2/(6+1), item_shop_city_category_sup_category=None,is_cnt=True):
    """
    Calculates EMA for specified groupings and returns metrics based on a given column type.
    
    Args:
        source (pd.DataFrame): Original data source.
        groupby (list of str): List of columns to group data by.
        alpha (float): EMA smoothing factor.
        item_shop_city_category_sup_category (pd.DataFrame): Datarame with unique [shop, item] pairs. Used to return data with all requierd [shop, item] pairs.
        is_cnt (bool): Whether to use item count data; set to False for price data.

    Returns:
        pd.DataFrame: Aggregated table with EMAs calculated for the specified groupings.
    """
    column='item_cnt_day' if is_cnt else 'item_price'
    means = create_pivot_table(source,index=groupby,item_shop_city_category_sup_category=item_shop_city_category_sup_category,column=column)
    #means = find_mean_by(source, groupby, item_shop_city_category_sup_category,is_cnt)
    if np.isclose(alpha , 1.0):
        numerical = [col for col in means.columns if col.isdigit()]
        numerical = [col for col in numerical if int(col) < 35]#include 34 as precise value calculated by this function

        train = means[['shop_id','item_id', *numerical]]
        return train
    
    emas = calculate_ema_6(means, alpha)
    
    numerical = [col for col in emas.columns if col.isdigit()]
    numerical = [col for col in numerical if int(col) < 35]

    train = emas[['shop_id','item_id', *numerical]]
    return train
    
def calculate_diff(df):
    """
    Calculates time difference since first occurrence for each item in each month.
    
    Args:
        df (pd.DataFrame): Data table with month columns for calculating time differences.

    Returns:
        pd.DataFrame: Data table with calculated time difference since first sale.
    """
    df_1=df.copy()

    df['-1'] = 0.0

    for dbn in range(0,35):
        month_columns = [str(i) for i in range(-1,dbn+1)]
        
        
        first_sale_months = df[month_columns].ge(0.00001).idxmax(axis=1).apply(int)#if mean < 0.00001 wouldnt work
        
        #print((first_sale_months == -1).sum())
        
        df_1[str(dbn)] =  int(dbn) -  first_sale_months.values 

        df_1.loc[first_sale_months == -1,month_columns ] = -1

    df=df.drop(['-1'], axis=1)
    df_1=df_1.drop(['-1'], axis=1)

    return df_1

def rename_columns(df, name=None):
    """
    Renames columns in a DataFrame inplace by appending a specified name as a prefix.
    
    Args:
        df (pd.DataFrame): Data to rename columns for.  
        name (str): Prefix to add to each column name.
    """
    new_columns = {}
    for col in df.columns:
        new_col = str(col)
        if new_col in ['shop_id','item_id','item_category_id','super_category','city']:
            continue

        else:
            #print(name)
            #print(new_col)
            new_col = name+'$'+new_col

        
        new_columns[col] = new_col
    
    #print(new_columns)
    df.rename(columns=new_columns, inplace=True)
    
    #return df



def merge_boosting(item_shops, names, chunksize=None,item_shop_city_category_sup_category=None):
    """
    Merges data from multiple sources into a single CSV file, handling large datasets in chunks.
    
    Args:
        item_shops (pd.DataFrame): Data with unique item-shop pairs.
        names (list of str): List of file names of files to read and merge data from.
        chunksize (int): Size of chunks for data processing.
        item_shop_city_category_sup_category (pd.DataFrame): Datarame with unique [shop, item] pairs. Used to write data with all requierd [shop, item] pairs.
    """
    def create_batch_for_writing(file_length, chunksize):
        """

        Args:
            file_length (int): _description_
            chunksize (int): _description_

        Returns:
            list[list[int,int]]: indexes for batches in files from panamesthes
        """
        l = file_length
        
        chunk_num = l // chunksize if l%chunksize==0  else   l // chunksize+ 1
        i_l = []
        for i in range(chunk_num):
            i_l.append([i*chunksize, min((i+1)*chunksize, file_length)])

        return i_l

    critical=['change','mean','ema']#cant calculate statistics for them for first 6 monthes (for change can, but we won't use it)

    #data = pd.read_csv('data/'+path+'.csv', chunksize=chunksize)
    idxs = create_batch_for_writing(len(item_shops), chunksize)

    first = True
    
    for idx in idxs:
        first_inner = True
        for path in names:
            
            batch=pd.read_csv(f'data/{path}.csv',skiprows=range(1,idx[0] + 1),nrows=idx[1] - idx[0])
            if first_inner:
                init = batch[['shop_id','item_id']]
                first_inner=False

            for i in range(FIRST_N_MONTH_TO_DROP):#exclude features where past can not be calculated
                
                if any(cr in path for cr in critical):
                    
                    batch=batch.drop(str(i), axis=1)
            
            rename_columns(batch, path)
            
            init = pd.merge(init, batch, on=['shop_id','item_id'], how = 'right')
            
            #del batch
            gc.collect()

        if first:
            
            init.merge(item_shop_city_category_sup_category,on=['shop_id','item_id'], how='left').to_csv('data/merged.csv', index=False)
            print('init memory usage,',sum(init.memory_usage()/10**6))
            first= False
        else:
            init.merge(item_shop_city_category_sup_category,on=['shop_id','item_id'], how='left').to_csv('data/merged.csv', mode='a', index=False, header=False)

    print('names in merged,',names )


def prepare_files(data_train, item_shop_city_category_sup_category, alpha=2/(6+1)):
    """
    Creates and saves files containing various featues.

    Created csv's will have columns [shop_id, item_id, *[str(i) for i in range(35)]]

    Note: item_cnt_day name is worng, it's actually item_cnt_month. Also item_price is a mean price of (shop_id, item_id) pairs in date_block_num
    Args:
        data_train (pd.DataFrame): Training data. Columns are shop_id, item_id, city, item_category_id, super_categoty, item_cnt_day, item_price
        item_shop_city_category_sup_category (pd.DataFrame): Datarame with unique [shop, item] pairs. Used to return data with all requierd [shop, item] pairs.
        alpha (float): EMA smoothing factor.

    Returns:
        list[str]: List of generated file names for the processed data. Used to merge them in a single csv
    """
    print('prepare_files started...')

    #list of columns to find EMA by. File name will have format ema_{'_'.join(gr)}.csv'. calculates EMA for last 6 monthes
    group_bys_EMA = [['shop_id','item_category_id'],
                 ['item_id','city'],
                 ['item_id'],
                 ['item_category_id','city'],
                 ['super_category'],
                 ['shop_id']]
    names = []
    for gr in group_bys_EMA:
        t1 = time.time()
        train = calculate_EMAs_pipeline(data_train,
                                         gr, 
                                         alpha=alpha,
                                         item_shop_city_category_sup_category=item_shop_city_category_sup_category,
                                         is_cnt=True)
        #print(train)
        train.to_csv(f'data/ema_{'_'.join(gr)}.csv', index=False)
        t2 = time.time()
        print(f'EMA calculated for {'_'.join(gr)}.csv; time:', t2-t1)
        names.append(f'ema_{'_'.join(gr)}')
        
    #list of lists of columns to find sales by. File name will have format value_{'_'.join(gr)}.csv'. This calculates sales for 1 date_block_nnum
    group_bys_lags = [['shop_id','item_id'],
                 ['item_id'],
                 ['item_category_id']
                 ]
    
    for gr in group_bys_lags:
        t1 = time.time()
        train = calculate_EMAs_pipeline(data_train,
                                         gr, 
                                         alpha=1.0,
                                         item_shop_city_category_sup_category=item_shop_city_category_sup_category,
                                         is_cnt=True)
        #print(train)
        
        train.to_csv(f'data/value_{'_'.join(gr)}.csv', index=False)
        t2 = time.time()
        print(f'EMA calculated for {'_'.join(gr)}.csv; time:', t2-t1)
        names.append(f'value_{'_'.join(gr)}')
    
    group_bys_lags_prices = [['item_id'],
                             ['shop_id']]
    
    for gr in group_bys_lags_prices:
        t1 = time.time()
        train = calculate_EMAs_pipeline(data_train,
                                         gr, 
                                         alpha=1.0,
                                         item_shop_city_category_sup_category=item_shop_city_category_sup_category,
                                         is_cnt=False)
        #print(train)
        
        train.to_csv(f'data/value_price_{'_'.join(gr)}.csv', index=False)
        t2 = time.time()
        print(f'EMA calculated for {'_'.join(gr)}.csv; time:', t2-t1)
        names.append(f'value_price_{'_'.join(gr)}')
    
    t1 = time.time()
    #prices = create_pivot_table(data_train,index=['item_id','shop_id'], item_shop_city_category_sup_category=item_shop_city_category_sup_category,column='item_price')
    mean_prices_items = calculate_EMAs_pipeline(data_train, #Worng or not already
                                                ['item_id'], 
                                                item_shop_city_category_sup_category=item_shop_city_category_sup_category,
                                                alpha=1.0,
                                                is_cnt=False)
    
    changes = create_change(mean_prices_items)
    changes.to_csv('data/item_price_change.csv', index=False)
    t2 = time.time()
    print(f'price change calculated; time:', t2-t1)
    names += ['item_price_change']

    t1 = time.time()
    value_item_sales = pd.read_csv('data/value_item_id.csv')
    diff=calculate_diff(value_item_sales)
    diff.to_csv('data/item_dbn_diff.csv', index=False)
    t2 = time.time()
    print(f'dbn diff calculated; time:', t2-t1)
    names += ['item_dbn_diff']
    return names


def calc_and_write_chunk(data_train,item_shop_city_category_sup_category, chunksize_when_writing):

    """
    Calculates and writes EMA metrics and other statistics in chunks to handle large data sizes.
    
    Args:
        data_train (pd.DataFrame): Training data. Columns are shop_id, item_id, city, item_category_id, super_categoty, item_cnt_day, item_price
        item_shop_city_category_sup_category (pd.DataFrame): Datarame with unique [shop, item] pairs. Used to calculate and write data with all requierd [shop, item] pairs.
        chunksize_when_writing (int): Chunk size for reading data from created csv's and writing data into common csv
    """

    alpha=0.0
    print('prepare_files started..')
    names = prepare_files(data_train,item_shop_city_category_sup_category, alpha=alpha)
    print('prepare_files finished')
    
    prepare_for_boosting=True
    
    shop_item=item_shop_city_category_sup_category[['shop_id','item_id']].drop_duplicates()
    print('merge started')
    t1=time.time()
    if prepare_for_boosting:
        merge_boosting(shop_item, names, chunksize_when_writing,item_shop_city_category_sup_category)
    t2=time.time()
    print('merge ended, time:', t2-t1)
    
def create_split(past, chunk_size=300000):
    """
    Creates randomized splits of indices for batching data preparation.

    Args:
        past (array-like): Dataset to split.
        chunk_size (int): Number of samples per batch.

    Returns:
        list of arrays: List of index arrays for each batch.
    """
    l = len(past)
    idxs = np.random.permutation(l)
    chunk_num = l // chunk_size if l%chunk_size==0  else   l // chunk_size+ 1
    i_l = []
    for i in range(chunk_num):
        i_l.append(idxs[i*chunk_size:(i+1)*chunk_size])

    return i_l

def prepare_batches(past,item_shop_city_category_sup_category, chunk_size):
    """
    Prepares batches of Cartesian product data with merged metadata for processing.

    Args:
        past (array-like): Data representing all unique shop-item pairs.
        item_shop_city_category_sup_category (pd.DataFrame): Datarame with unique [shop, item] pairs and relevant colymns
        chunk_size (int): Size of each batch.

    Yields:
        pd.DataFrame: Batch of merged data(only item,shop,city,category,sup_category columns)
    """
    l = len(past)
    idxs = create_split(past, chunk_size=chunk_size)
    for chunk in idxs:
        all_cartesians = past[chunk]
        all_cartesians=pd.DataFrame({'shop_id':all_cartesians[:,0], 'item_id':all_cartesians[:,1]})


        shop_city=item_shop_city_category_sup_category[['shop_id','city']].drop_duplicates()
        item_category=item_shop_city_category_sup_category[['item_id','item_category_id']].drop_duplicates()
        category_super_category=item_shop_city_category_sup_category[['item_category_id','super_category']].drop_duplicates()


        item_shop_city_category_sup_category_ret = all_cartesians.merge(shop_city).merge(item_category).merge(category_super_category)
        yield item_shop_city_category_sup_category_ret

   
def parse_city(shop_name):
    """
    Extracts city name from shop name string.
    
    Args:
        shop_name (str): Name of the shop.

    Returns:
        str: Extracted city name.
    """

    if shop_name.split()[0] == '!Якутск':
        return  'Якутск'

    if shop_name.split()[0] == 'Сергиев':
            return  'Сергиев Посад'
    else:
        return shop_name.split()[0]

def create_item_shop_city_category_super_category(shop_city_pairs, merged):
    """
    Merges shop and item data with additional city, category, and super-category information.
    
    Args:
        shop_city_pairs (array-like): Array of unique shop-item pairs.
        merged (pd.DataFrame): Data containing shop and item metadata.

    Returns:
        pd.DataFrame: Merged table with city, item category, and super-category columns.
    """
    shop_city_pairs=pd.DataFrame({'shop_id':shop_city_pairs[:,0],'item_id':shop_city_pairs[:,1]})
    merged['city'] = merged['shop_name'].apply(parse_city).astype('category').cat.codes.astype(np.uint8)
    
    shop_city = merged.groupby('shop_id')['city'].unique().reset_index()
    shop_city_pairs=shop_city_pairs.merge(shop_city, how='left')

    item_category = merged.groupby('item_id')['item_category_id'].unique().reset_index()
    shop_city_pairs=shop_city_pairs.merge(item_category, how='left')
    shop_city_pairs['item_category_id'] = shop_city_pairs['item_category_id'].apply(lambda a:a[0])

    merged['super_category'] = merged['item_category_name'].apply(lambda a: a.split()[0]).astype('category').cat.codes.astype(np.uint8)
    category_super_category = merged.groupby('item_category_id')['super_category'].unique().reset_index()
    shop_city_pairs=shop_city_pairs.merge(category_super_category, how='left')
    shop_city_pairs['super_category'] = shop_city_pairs['super_category'].apply(lambda a:a[0])
    
    return merged[['shop_id','item_id','item_category_id','city','super_category']]

if __name__ == '__main__':
    total_time1 = time.time()
    data_train = pd.read_csv('../data_cleaned/data_train.csv')
    item_categories = pd.read_csv('../data_cleaned/item_categories.csv')
    items = pd.read_csv('../data_cleaned/items.csv')
    shops = pd.read_csv('../data_cleaned/shops.csv')
    test = pd.read_csv('../data_cleaned/test.csv')
    test['date_block_num'] = 34
    data_train = data_train.merge(test, on=['item_id','shop_id','date_block_num'], how='outer').fillna(0)


    print('IDs preparation started...')
    t1 = time.time()
    past = prepare_past_ID_s_CARTESIAN(data_train)[-1][-1]
    t2 = time.time()
    print('IDs preparation time:', t2-t1)


    t1=time.time()
    grouped=data_train.groupby(['shop_id','item_id','date_block_num']).agg({'item_price':'mean',
                                                                    'item_cnt_day':'sum'
                                                                    })
    #Note that there is no rename of item_cnt_day and item_price columns
    grouped=grouped.reset_index()
    grouped = grouped.sort_values(by='date_block_num')
    merged = grouped.merge(items, how='left').merge(item_categories, how='left').merge(shops, how='left')
    item_shop_city_category_sup_category = create_item_shop_city_category_super_category(past, merged)
    t2=time.time()
    print('creating item_shop_city_category_sup_category time:',t2-t1)

    data_train = merged.sort_values(by='date_block_num')
    
    del test
    
    first = True
    chunk_size=len(past)#if chunk_size < len(past), statistics may be misleading
    chunksize_when_writing = 300000
    
    for item_shop_city_category_sup_category in prepare_batches(past,item_shop_city_category_sup_category, chunk_size=chunk_size):
        calc_and_write_chunk(data_train,item_shop_city_category_sup_category, chunksize_when_writing)
        first = False
    total_time2 = time.time()
    print('total data preparation time,',total_time2-total_time1)