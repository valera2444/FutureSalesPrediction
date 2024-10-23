import numpy as np
import pandas as pd

import seaborn as sns
from sklearn.preprocessing import TargetEncoder
import gc
import sys
import calendar

import warnings


def sizeof_fmt(num, suffix='B'):
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)
    
   
def parse_city(shop_name):
    if shop_name.split()[0] == '!Якутск':
        return  'Якутск'

    if shop_name.split()[0] == 'Сергиев':
            return  'Сергиев Посад'
    else:
        return shop_name.split()[0]



def number_to_month(numbers):
    # Create a list of month names using the calendar module
    month_names = list(calendar.month_name)[1:]  # Extract months from 1 to 12 (exclude the empty element at index 0)
    
    # Convert the numbers to month names
    result = [month_names[num % 12] for num in numbers]  # Use num % 12 to handle numbers beyond 11
    
    return result


def number_to_month(numbers):
    result = [num %12 for num in numbers]  # Use num % 12 to handle numbers beyond 11
    
    return result

def convert_to_year(month_num):
    # January 2013 is the starting point
    start_year = 2013
    
    # Calculate the year
    year = start_year + (month_num // 12)
    
    return year


def create_lags(data,item_lags=None,price_lags=None, test_included=False):
    #if test_included:
    #    lagged_items = data[data['date_block_num'] < 34] [['date_block_num','item_cnt_month','shop_id','item_id']].copy()
    #    lagged_prices = data[data['date_block_num'] < 34] [['date_block_num','avg_item_price','shop_id','item_id']].copy()
    #else:
    #    lagged_items = data[['date_block_num','item_cnt_month','shop_id','item_id']].copy()
    #    lagged_prices = data[['date_block_num','avg_item_price','shop_id','item_id']].copy()
    lagged_items = data[['date_block_num','item_cnt_month','shop_id','item_id']]
    lagged_prices = data[['date_block_num','avg_item_price','shop_id','item_id']]
    for lag in item_lags:
        lagged_items.loc[:,'date_block_num']+=lag#previous month becomes present
        data=data.merge(lagged_items, how='left',on=['date_block_num','shop_id','item_id'], suffixes=('', f'_lag_{lag}'))
        lagged_items.loc[:,'date_block_num']-=lag

    
    for lag in price_lags:
        lagged_prices.loc[:,'date_block_num']+=lag#previous month becomes present
        data=data.merge(lagged_prices, how='left',on=['date_block_num','shop_id','item_id'], suffixes=('', f'_lag_{lag}'))
        lagged_prices.loc[:,'date_block_num']-=lag

        
    return data






def create_lags_columns(data, columns,item_lags=[1,2,3,4,5,6],price_lags=[1,2,3,4,5,6], test_included=False):#date_block_num required as first argument
    #if test_included:
    #    lagged_items = data [data['date_block_num'] < 34]  [[*columns,'item_cnt_month']] 
    #    lagged_prices = data [data['date_block_num'] < 34]  [[*columns,'avg_item_price']]
    #else:
    lagged_items = data[[*columns,'item_cnt_month']]
    lagged_prices = data[[*columns,'avg_item_price']]
    
    for lag in item_lags:
        lagged_items.loc[:,'date_block_num']+=lag#previous month becomes present

        averaged = lagged_items.groupby(columns).mean().reset_index()#future 
        
        data=data.merge(averaged, how='left',on=columns, suffixes=('', f'{'_'.join(columns[1:])}_lag_{lag}'))
        
        lagged_items.loc[:,'date_block_num']-=lag

    
    for lag in price_lags:
        lagged_prices.loc[:,'date_block_num']+=lag#previous month becomes present

        averaged = lagged_prices.groupby(columns).mean().reset_index()
        
        data=data.merge(averaged, how='left',on=columns, suffixes=('', f'{'_'.join(columns[1:])}_lag_{lag}'))
        lagged_prices.loc[:,'date_block_num']-=lag

        
    return data

def calculate_ema_3(df, target='item_cnt_month', columns=None, test_included=True, alpha=2/(3+1)):
    weights = (1 - alpha) ** np.arange(3)

    # Define the columns for lags
    lag_cols = [f'{target}{'_'.join(columns[1:])}_lag_{i}' for i in range(1, 4)]

    # Replace NaNs with zeros in the lags
    df_in = df[lag_cols].fillna(0)

    # Calculate weighted sums and weights
    weighted_sums = np.dot(df_in[lag_cols].values, weights)
    valid_weights = (df_in[lag_cols] != 0).values.dot(weights)

    # Avoid division by zero
    valid_weights[valid_weights == 0] = np.nan

    # Calculate EMA
    df[f'ema_3_{target}_{'_'.join(columns[1:])}'] = weighted_sums / valid_weights

    return df

#alpha == 0 is simple mean
def calculate_ema_6(df, min_periods=3, target='item_cnt_month', columns=None, test_included=True, alpha=2/(6+1)):
    weights = (1 - alpha) ** np.arange(6)
    print(weights)
    # Define the columns for lags
    lag_cols = [f'{target}{'_'.join(columns[1:])}_lag_{i}' for i in range(1, 7)]

    # Replace NaNs with zeros in the lags
    df_in = df[lag_cols].fillna(0)

    # Calculate weighted sums and weights
    weighted_sums = np.dot(df_in[lag_cols].values, weights)
    #valid_weights = (df_in[lag_cols] != 0).values.dot(weights)

    # Avoid division by zero
    #print(valid_weights)
    #valid_weights[valid_weights == 0] = np.nan

    # Calculate EMA
    df[f'ema_6_{target}_{'_'.join(columns[1:])}'] = weighted_sums / sum(weights)
    df[f'ema_6_{target}_{'_'.join(columns[1:])}']=df[f'ema_6_{target}_{'_'.join(columns[1:])}'].fillna(0)
    return df

def create_last_seen(data):#Takes some minutes
    prevs=data.groupby(['item_id'])['date_block_num'].unique().rename('blocks')
    mer = data.merge(prevs, how='left', on='item_id')
    for idx in range(len(mer)):
        curr =  mer.loc[idx,'blocks'] 
        
        prevs = np.array(curr) < mer.loc[idx,'date_block_num']
    
        if sum(prevs) == 0:
            mer.loc[idx,'blocks'] = -1
        else:
            mer.loc[idx,'blocks'] = curr [prevs] .max()

    data['last_seen'] = mer['blocks']
    return data


def find_means_last6(lagged, columns=None):#date_block_num required as first argument
    print(list(lagged.columns))
    lagged=lagged.fillna(0)
    means = lagged[['item_cnt_month_lag_1', 
                 'item_cnt_month_lag_2',
                 'item_cnt_month_lag_3',
                 'item_cnt_month_lag_4', 
                 'item_cnt_month_lag_5', 
                 'item_cnt_month_lag_6'
                   ]].mean(axis=1).rename('avg')

    
   
    m_i = pd.concat([lagged[columns], means],axis=1)
    m_i = m_i.groupby(columns)['avg'].transform('mean')
    lagged[f'mean_item_cnt_6month{'_'.join(columns[1:])}'] = m_i
    return lagged

import time
def create_EMAs(df,cols, test_included=True, alpha=None):
    df = create_lags_columns(df,columns=cols,test_included=test_included )
    start = time.time()
    #df = calculate_ema_3(df, target='item_cnt_month', columns=cols,test_included=test_included, alpha=alpha )
    #df = calculate_ema_3(df, target='avg_item_price', columns=cols,test_included=test_included ,alpha=alpha)
    df = calculate_ema_6(df, target='item_cnt_month', columns=cols,test_included=test_included,alpha=alpha )
    #df = calculate_ema_6(df, target='avg_item_price', columns=cols,test_included=test_included )
    end = time.time()
    #print('memory before drop', df.memory_usage())
    
    df=df.drop([f'item_cnt_month{'_'.join(cols[1:])}_lag_1', 
                f'item_cnt_month{'_'.join(cols[1:])}_lag_2', 
                f'item_cnt_month{'_'.join(cols[1:])}_lag_3',
                f'avg_item_price{'_'.join(cols[1:])}_lag_1', 
                f'avg_item_price{'_'.join(cols[1:])}_lag_2', 
                f'avg_item_price{'_'.join(cols[1:])}_lag_3',
               ], axis=1)
    
    df=df.drop([f'item_cnt_month{'_'.join(cols[1:])}_lag_4', 
                f'item_cnt_month{'_'.join(cols[1:])}_lag_5', 
                f'item_cnt_month{'_'.join(cols[1:])}_lag_6',
                f'avg_item_price{'_'.join(cols[1:])}_lag_4', 
                f'avg_item_price{'_'.join(cols[1:])}_lag_5', 
                f'avg_item_price{'_'.join(cols[1:])}_lag_6',
               ], axis=1)
    
    #print('memory after drop', df.memory_usage())
    print('item_id EMA calculated , time,s:',end-start)
    return df

def transform(data_train, item_categories,items,shops, test_included=True ):
    grouped=data_train.groupby(['shop_id','item_id','date_block_num']).agg({'item_price':'mean',
                                                                    'item_cnt_day':'sum'
                                                                    })#take some (a lot)))) ) time
    
    for name, size in sorted(((name, sys.getsizeof(value)) for name, value in list(
                              locals().items())), key= lambda x: -x[1])[:10]:
        print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))
        
    print('1st grouping done')
    
    grouped=grouped.reset_index()
    grouped = grouped.sort_values(by='date_block_num')
    merged = grouped.merge(items, how='left').merge(item_categories, how='left').merge(shops, how='left')
    
    print('merged created')

   
    merged['month'] = np.array(number_to_month(merged['date_block_num'])).astype(np.uint8)
    merged['year'] = merged['date_block_num'].apply(convert_to_year).astype(np.uint16)
    merged['super_category'] = merged['item_category_name'].apply(lambda a: a.split()[0]).astype('category').cat.codes.astype(np.uint8)
    merged['city'] = merged['shop_name'].apply(parse_city).astype('category').cat.codes.astype(np.uint8)
    merged=merged.rename(columns={'item_price':'avg_item_price','item_cnt_day':'item_cnt_month'})
    
    
    ALPHA = 0.0
    merged['shop_id_cat'] = merged['shop_id'].astype(np.uint8)
    merged['item_id_cat'] = merged['item_id'].astype(np.uint16)
    merged['item_category_id_cat'] = merged['item_category_id'].astype(np.uint8)

    
    cols=['date_block_num','item_id']
    merged=create_EMAs(merged,cols, alpha = ALPHA)#168.3924057483673
    print(f'{cols} EMA calculated')
    
    
    cols=['date_block_num','item_id','city']#--------------
    merged=create_EMAs(merged,cols, alpha = ALPHA)
    print(f'{cols} EMA calculated')
    

    cols=['date_block_num','item_category_id_cat','shop_id']#--------------
    merged=create_EMAs(merged,cols, alpha = ALPHA)
    print(f'{cols} EMA calculated')
    
        
    cols=['date_block_num','item_category_id_cat','city']#------------34
    merged=create_EMAs(merged,cols, alpha = ALPHA)
    print(f'{cols} EMA calculated')

    #cols=['date_block_num','shop_id']#------------34
    #merged=create_EMAs(merged,cols)
    #print(f'{cols} EMA calculated')

    cols=['date_block_num','item_id','shop_id']#------------34
    merged=create_EMAs(merged,cols, alpha = ALPHA)
    print(f'{cols} EMA calculated')

    #merged = create_last_seen(merged)
    print('Last seen created')
    
    item_lags=list([*range(1,7)])
    price_lags=list([*range(1,2)])
    
    merged = create_lags(merged,item_lags=item_lags,price_lags=price_lags,test_included=test_included )#takes a bit time
    
    merged = create_lags_columns(merged, columns=['date_block_num','item_id'],item_lags=item_lags,price_lags=price_lags, test_included=False)
    merged = create_lags_columns(merged,  columns=['date_block_num','shop_id'],item_lags=item_lags,price_lags=price_lags, test_included=False)
    merged = create_lags_columns(merged, columns=['date_block_num','item_category_id_cat','shop_id'],item_lags=item_lags,price_lags=price_lags, test_included=False)
    merged = create_lags_columns(merged,  columns= ['date_block_num','item_category_id_cat'],item_lags=item_lags,price_lags=price_lags, test_included=False)
    
    print('laggs created')
    
    
    
   # merged = find_means_last6(merged, ['date_block_num','item_id','shop_id',])
   # merged = find_means_last6(merged, ['date_block_num','item_id'])
   # merged = find_means_last6(merged, ['date_block_num','item_category_id','shop_id'])
   # merged = find_means_last6(merged, ['date_block_num','item_category_id','city'])
   # merged = find_means_last6(merged, ['date_block_num','item_category_id'])
   

    
    
    print('means for last 6 mothes found')
    
    merged['first_date_block_num'] = merged.groupby('item_id')['date_block_num'].transform('min')
    merged['date_block_num_diff'] = merged['date_block_num'] - merged['first_date_block_num']
    
    
    
    return merged#merged['date_block_num' >= 3] - for lag-based learning()


if __name__ == '__main__':

    warnings.simplefilter(action='ignore', category=FutureWarning)
    data_train = pd.read_csv('../data_cleaned/data_train.csv')
    item_categories = pd.read_csv('../data_cleaned/item_categories.csv')
    items = pd.read_csv('../data_cleaned/items.csv')
    shops = pd.read_csv('../data_cleaned/shops.csv')
    test = pd.read_csv('../data_cleaned/test.csv')
    test['date_block_num'] = 34
    transformed = transform(pd.concat([data_train,test]),  item_categories,items,shops,test_included=True )
    transformed.to_csv('../data.csv')
