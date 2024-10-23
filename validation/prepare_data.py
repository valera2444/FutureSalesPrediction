import numpy as np
import pandas as pd

import gc


FIRST_N_MONTH_TO_DROP = 6

def create_change(table):
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
def create_lag_table_general(data,column='item_cnt_monthitem_id_lag_1',
                             whole=None,
                             groupby = ['item_id'],
                             item_shop_city_category_sup_category=None):#long - ~467s
    #whole contains lagged, which means for dbn = 24 lag_1 calculated on 23
    #(item cnt month for dbn==24 stored in lag_1 for dbn == 25 )
    #to make whole same format as data, we need to whole['date_block_num']-=1
    start = time.time()

    whole=whole[['shop_id','item_id','date_block_num',column]]
    #new_table = data.copy()
    if 'lag' in column:
        whole['date_block_num'] -= 1
        whole = whole[whole['date_block_num'] >= 0]

    

    #print(whole.date_block_num)
    agg = 'mean'
    df = pd.pivot_table(whole, 
                    index=['shop_id','item_id'],
                    columns=['date_block_num'],
                    values=[column],
                    aggfunc=agg, 
                    fill_value=0
                   ).reset_index()

    train = pd.DataFrame({col:df[col] for col in ['shop_id','item_id']})
    
    col = df[column]
    del df
    gc.collect()
    train = pd.concat([train,col], axis=1)

    numerical = [col for col in train.columns if isinstance(col, int)]
    numerical = [col for col in numerical if col < 34]
    train = item_shop_city_category_sup_category.merge(train, how='left').fillna(0)
    train = train[['shop_id','item_id', *numerical]]
    print(train)
    return train


def create_item_shop_data(data, column='item_cnt_day'):
    if column == 'item_cnt_day':
        agg = 'sum'
    else:
        agg='mean'
    df = pd.pivot_table(data, 
                    index=['shop_id','item_id'],
                    columns=['date_block_num'],
                    values=[column],
                    aggfunc=agg, 
                    fill_value=0
                   ).reset_index()
   # print(df)
    train = pd.DataFrame({'shop_id':df['shop_id'], 'item_id':df['item_id']})
    
    item_cnt_day = df[column]
    train = pd.concat([train,item_cnt_day], axis=1)
    #print(items)
    merged = train.merge(items, how='left').merge(item_categories, how='left').merge(shops, how='left') \
        .drop(['item_name','item_category_name','shop_name'], axis=1)


    merged.columns = [str(col) for col in merged.columns]
    return merged



def write_file_by_name(merged,column=None, whole=None, groupby='', shop_city=None,category_super_category=None ):
    time1 = time.time()
    item_shop_city_category_sup_category = merged.merge(shop_city). \
    merge(category_super_category)[['shop_id',	'item_id','item_category_id','city','super_category']]
    
    item_shop_city_category_sup_category['city'] = item_shop_city_category_sup_category['city'].map(lambda a:a[0])
    item_shop_city_category_sup_category['super_category'] = item_shop_city_category_sup_category['super_category'].map(lambda a:a[0])

    table_ema6_item = create_lag_table_general(merged,
                                               column=column,
                                               whole=whole,
                                               groupby=groupby, 
                                               item_shop_city_category_sup_category=item_shop_city_category_sup_category)
    
    table_ema6_item.to_csv(f'data/{column}.csv', index=False)
    #print(table_ema6_item.columns)
    del table_ema6_item
    gc.collect()

    print('time taken for',column,':', time.time() - time1  )





names_base = ['ema_6_item_cnt_month_item_id',
             'ema_6_item_cnt_month_item_id_shop_id',
             'ema_6_item_cnt_month_item_category_id_cat_city',
             'ema_6_item_cnt_month_item_category_id_cat_shop_id',
             'date_block_num_diff',
            'avg_item_priceitem_id_lag_1',
            'item_cnt_monthitem_category_id_cat_lag_1',
            'item_cnt_month_lag_1',
             'avg_item_price']
#Lag in a name left from columns name; Value for curr month stored in validation/data

def rename_columns(df, name=None):
    
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


def merge_boosting(merged):

    def merge_csvs_boosting(df0, names):
        critical=['change','mean','ema']
        rename_columns(df0, 'shop_item_cnt')
        for df_name in names:
            df = pd.read_csv('data/'+df_name+'.csv')
            print(sum(df.memory_usage()/10**6))
    
            
            for i in range(FIRST_N_MONTH_TO_DROP):#exclude features where past can not be calculated
                
                if any(cr in df_name for cr in critical):
                    
                    df=df.drop(str(i), axis=1)
            
            rename_columns(df, df_name)
            
            df0 = pd.merge(df0, df, on=['shop_id','item_id'], how = 'left')
            del df
            gc.collect()
    
        return df0
    
    names_boosting =['ema_6_item_cnt_month_item_id',
                     #'ema_6_item_cnt_month_item_category_id_cat_city',
                     'ema_6_item_cnt_month_item_category_id_cat_shop_id',
                     'date_block_num_diff',
                     'avg_item_priceitem_id_lag_1',
                     'item_cnt_monthitem_category_id_cat_lag_1',
                     'item_cnt_monthitem_id_lag_1',
                     #'avg_item_price',
                     'item_price_change']
    
    
    merge_csvs_boosting(merged, names_boosting).to_csv('data/merged.csv', index=False)


def merge_LSTM(merged):

    def merge_csvs_LSTM(df0, names):
        
        df0=df0.drop('0', axis=1)
        
        rename_columns(df0, 'shop_item_cnt')
        for df_name in names:
            df = pd.read_csv('data/'+df_name+'.csv')
            print(sum(df.memory_usage()/10**6))
    
            df=df.drop(str(0), axis=1)
        
            rename_columns(df, df_name)
            
            df0 = pd.merge(df0, df, on=['shop_id','item_id'], how = 'left')
            del df
            gc.collect()
    
        return df0
    
    names_LSTM = ['avg_item_price']
    merge_csvs_LSTM(merged, names_LSTM).to_csv('data/merged.csv', index=False)

def prepare_files(merged, whole,shop_city,category_super_category):
    
    write_file_by_name(merged,column='ema_6_item_cnt_month_item_id',
                        whole=whole, 
                        groupby=['item_id'],
                        shop_city=shop_city,
                        category_super_category=category_super_category
                        )
    #write_file_by_name(merged,column='ema_6_item_cnt_month_item_id_shop_id', whole=whole, groupby=['item_id','shop_id'])
    write_file_by_name(merged,
                       column='ema_6_item_cnt_month_item_category_id_cat_city', 
                       whole=whole, groupby= ['item_category_id','city'],
                       shop_city=shop_city,
                        category_super_category=category_super_category
                        )
    

    write_file_by_name(merged,
                       column='ema_6_item_cnt_month_item_category_id_cat_shop_id', 
                       whole=whole, 
                       groupby=['item_category_id','shop_id'],
                       shop_city=shop_city,
                        category_super_category=category_super_category
                        )
    
    write_file_by_name(merged,
                       column='date_block_num_diff', 
                       whole=whole, 
                       groupby=['item_id'],
                       shop_city=shop_city,
                        category_super_category=category_super_category
                        )
    write_file_by_name(merged,column='avg_item_priceitem_id_lag_1', 
                       whole=whole, 
                       groupby=['item_id'],
                       shop_city=shop_city,
                        category_super_category=category_super_category
                        )
    
    write_file_by_name(merged,
                       column='item_cnt_monthitem_category_id_cat_lag_1', 
                       whole=whole, 
                       groupby=['item_category_id'],
                       shop_city=shop_city,
                        category_super_category=category_super_category
                        )
    
    write_file_by_name(merged,
                       column='avg_item_price',
                         whole=whole, 
                         groupby=['item_id','shop_id'],
                         shop_city=shop_city,
                        category_super_category=category_super_category
                        )
    
    write_file_by_name(merged,
                       column='item_cnt_monthitem_id_lag_1',
                         whole=whole, 
                         groupby=['item_id'],
                         shop_city=shop_city,
                        category_super_category=category_super_category
                        )
    
    
    
    table_item_price = pd.read_csv('data/avg_item_priceitem_id_lag_1.csv')
    price_changes_items = create_change(table_item_price)
    price_changes_items.to_csv('data/item_price_change.csv', index=False)
    del table_item_price
    del price_changes_items
    gc.collect()
    
    prices = pd.read_csv('data/avg_item_price.csv')
    price_changes = create_change(prices)
    price_changes.to_csv('data/shop_item_price_change.csv', index=False)
    del price_changes
    del prices
    gc.collect()
    
    cnt_changes_items_shops = create_change(merged)
    cnt_changes_items_shops.to_csv('data/cnt_changes_items_shops.csv', index=False)
    del cnt_changes_items_shops
    gc.collect()
    
    
    table_item_cnt = pd.read_csv('data/item_cnt_monthitem_id_lag_1.csv')
    cnt_changes_items = create_change(table_item_cnt)
    cnt_changes_items.to_csv('data/cnt_changes_items.csv', index=False)
    del cnt_changes_items
    del table_item_cnt
    gc.collect()
    

if __name__ == '__main__':

    data_train = pd.read_csv('../data_cleaned/data_train.csv')
    item_categories = pd.read_csv('../data_cleaned/item_categories.csv')
    items = pd.read_csv('../data_cleaned/items.csv')
    shops = pd.read_csv('../data_cleaned/shops.csv')
    test = pd.read_csv('../data_cleaned/test.csv')
    test['date_block_num'] = 34
    data_train = data_train.merge(test, on=['item_id','shop_id','date_block_num'], how='outer').fillna(0)
    
    
    data_train=data_train.reset_index()
    data_train = data_train.sort_values(by='date_block_num')
    train = data_train.copy()
    whole = pd.read_csv('../data.csv')
    
    shop_city = whole.groupby('shop_id')['city'].unique().reset_index()
    
    category_super_category = whole.groupby('item_category_id')['super_category'].unique().reset_index()
    
    
    merged = create_item_shop_data(data_train)

    
    #prepare_files(merged,whole,shop_city,category_super_category)
    
    merged=merged.merge(shop_city, on='shop_id', how='left')
    merged=merged.merge(category_super_category, on='item_category_id', how='left')
    merged['city'] = merged['city'].map(lambda a:a[0])
    merged['super_category'] = merged['super_category'].map(lambda a:a[0])

    prepare_for_boosting=True
    
    if prepare_for_boosting:
        merge_boosting(merged)

    else:
        merge_LSTM(merged)