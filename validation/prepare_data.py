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

def calculate_ema_6(table, alpha=2/(6+1)):
    EMAs = pd.DataFrame(columns = table.columns)
    EMAs['shop_id'] = table['shop_id']
    EMAs['item_id'] = table['item_id']
    EMA_period = 6
    weights = (1 - alpha) ** np.arange(6)
    print(weights)
    
    for dbn in range(EMA_period, 34):
        lag_cols =  [str(i) for i in range(dbn - EMA_period+1, dbn+1)]
        
        df_in = table[lag_cols].fillna(0)
        
        weighted_sums = np.dot(df_in[lag_cols].values, weights)
        
        EMAs[str(dbn)] = weighted_sums / sum(weights)
    
    return EMAs

def find_mean_by(merged, group_by=None,item_shop_city_category_sup_category=None):
    merged=merged.merge(item_shop_city_category_sup_category, how='left')
    means = merged.groupby(group_by)[[name for name in merged.columns if name.isdigit()]].mean().reset_index()
    return item_shop_city_category_sup_category.merge(means, how='left')

def calculate_EMAs_pipeline(source, groupby, alpha=2/(6+1), item_shop_city_category_sup_category=None):
    
    means = find_mean_by(source, groupby, item_shop_city_category_sup_category)
    if np.isclose(alpha , 1.0):
        numerical = [col for col in means.columns if col.isdigit()]
        numerical = [col for col in numerical if int(col) < 34]

        train = means[['shop_id','item_id', *numerical]]
        return train
    
    emas = calculate_ema_6(means, alpha)
    
    numerical = [col for col in emas.columns if col.isdigit()]
    numerical = [col for col in numerical if int(col) < 34]

    train = emas[['shop_id','item_id', *numerical]]
    return train
    


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


def merge_boosting(merged, pathes):
    group_bys_EMA = [['shop_id','item_category_id'],
                    ['item_id','city'],
                    ['item_id'],
                    ['item_category_id','city']]
    EMA_names = [f'data/ema_{'_'.join(gr)}' for gr in group_bys_EMA]

    group_bys_lags = [
        #['shop_id','item_id'],
        ['item_id'],
        ['shop_id']
        ]
    lags_names = [f'data/value_{'_'.join(gr)}' for gr in group_bys_lags]
    
    
    def merge_csvs_boosting(df0, pathes):
        critical=['change','mean','ema']
        rename_columns(df0, 'shop_item_cnt')
        for path in pathes:
            df = pd.read_csv('data/'+path+'.csv')
            print(sum(df.memory_usage()/10**6))
    
            
            for i in range(FIRST_N_MONTH_TO_DROP):#exclude features where past can not be calculated
                
                if any(cr in path for cr in critical):
                    
                    df=df.drop(str(i), axis=1)
            
            rename_columns(df, path)
            
            df0 = pd.merge(df0, df, on=['shop_id','item_id'], how = 'left')
            del df
            gc.collect()
    
        return df0
    
    
    
    
    merge_csvs_boosting(merged, pathes).to_csv('data/merged.csv', index=False)
    print('names in merged,',pathes )


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
    
    names_LSTM = []
    merge_csvs_LSTM(merged, names_LSTM).to_csv('data/merged.csv', index=False)

def prepare_files(merged,data_train, item_shop_city_category_sup_category, alpha=2/(6+1)):
    group_bys_EMA = [['shop_id','item_category_id'],
                 ['item_id','city'],
                 ['item_id'],
                 ['item_category_id','city']]
    names = []
    for gr in group_bys_EMA:
        train = calculate_EMAs_pipeline(merged,
                                         gr, 
                                         alpha=alpha,
                                         item_shop_city_category_sup_category=item_shop_city_category_sup_category)
        print(train)
        train.to_csv(f'data/ema_{'_'.join(gr)}.csv', index=False)
        names.append(f'ema_{'_'.join(gr)}')
        
    group_bys_lags = [['shop_id','item_id'],
                 ['item_id'],
                 ['shop_id']]
    
    for gr in group_bys_lags:
        train = calculate_EMAs_pipeline(merged,
                                         gr, 
                                         alpha=1.0,
                                         item_shop_city_category_sup_category=item_shop_city_category_sup_category)
        print(train)
        train.to_csv(f'data/value_{'_'.join(gr)}.csv', index=False)
        names.append(f'value_{'_'.join(gr)}')


    prices = create_item_shop_data(data_train, column='item_price')
    mean_prices_items = calculate_EMAs_pipeline(prices, 
                                                ['item_id'], 
                                                item_shop_city_category_sup_category=item_shop_city_category_sup_category,
                                                alpha=1.0)
    changes = create_change(mean_prices_items)
    changes.to_csv('data/item_price_change.csv', index=False)
    names += ['item_price_change']
    return names

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

    merged = create_item_shop_data(data_train)
    #may be would be better to merge with (shop, item) cartesian
    shop_city = whole.groupby('shop_id')['city'].unique().reset_index()
    
    category_super_category = whole.groupby('item_category_id')['super_category'].unique().reset_index()
    item_shop_city_category_sup_category = merged.merge(shop_city). \
    merge(category_super_category)[['shop_id',	'item_id','item_category_id','city','super_category']]

    item_shop_city_category_sup_category['city'] = item_shop_city_category_sup_category['city'].map(lambda a:a[0])
    item_shop_city_category_sup_category['super_category'] = item_shop_city_category_sup_category['super_category'].map(lambda a:a[0])
    
    
    del whole
    pathes = prepare_files(merged,data_train,item_shop_city_category_sup_category, alpha=0.0)
    
    merged=merged.merge(shop_city, on='shop_id', how='left')
    merged=merged.merge(category_super_category, on='item_category_id', how='left')
    merged['city'] = merged['city'].map(lambda a:a[0])
    merged['super_category'] = merged['super_category'].map(lambda a:a[0])

    prepare_for_boosting=True
    
    if prepare_for_boosting:
        merge_boosting(merged, pathes)

    else:
        merge_LSTM(merged)