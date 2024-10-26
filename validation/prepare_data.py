import numpy as np
import pandas as pd

import gc


FIRST_N_MONTH_TO_DROP = 6

def prepare_past_ID_s_CARTESIAN(data_train):
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
                             cartesians[block], axis=0)#shop_item_pairs_WITH_PREV_in_dbn doesnt contain 34 month
        
        shop_item_pairs_WITH_PREV_in_dbn[block] = np.unique(arr, axis=0)
        print(len(shop_item_pairs_WITH_PREV_in_dbn[block]))
    return shop_item_pairs_in_dbn, shop_item_pairs_WITH_PREV_in_dbn

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
    
    for dbn in range(EMA_period, 35):#???
        lag_cols =  [str(i) for i in range(dbn - EMA_period+1, dbn+1)]
        
        df_in = table[lag_cols].fillna(0)
        
        weighted_sums = np.dot(df_in[lag_cols].values, weights)
        
        EMAs[str(dbn)] = weighted_sums / sum(weights)
    
    return EMAs

def find_mean_by(merged, group_by=None,item_shop_city_category_sup_category=None, is_cnt=True):
    merged=merged.merge(item_shop_city_category_sup_category, how='left')
    if is_cnt:
        means = merged.groupby(group_by)[[name for name in merged.columns if name.isdigit()]].mean().reset_index()

    else:
        means = merged.groupby(group_by)[[name for name in merged.columns if name.isdigit()]].mean().reset_index()

    return item_shop_city_category_sup_category.merge(means, how='left')

def calculate_EMAs_pipeline(source, groupby, alpha=2/(6+1), item_shop_city_category_sup_category=None,is_cnt=True):
    
    means = find_mean_by(source, groupby, item_shop_city_category_sup_category,is_cnt)
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



def merge_boosting(item_shops, pathes, chunksize=None,item_shop_city_category_sup_category=None):

    def create_batch_for_writing(file_length, chunksize):
        l = file_length
        
        chunk_num = l // chunksize if l%chunksize==0  else   l // chunksize+ 1
        i_l = []
        for i in range(chunk_num):
            i_l.append([i*chunksize, min((i+1)*chunksize, file_length)])

        return i_l

    critical=['change','mean','ema']

    #data = pd.read_csv('data/'+path+'.csv', chunksize=chunksize)
    idxs = create_batch_for_writing(len(item_shops), chunksize)

    first = True
    
    for idx in idxs:
        first_inner = True
        for path in pathes:
            
            batch=pd.read_csv(f'data/{path}.csv',skiprows=range(1,idx[0] + 1),nrows=idx[1] - idx[0])
            if first_inner:
                init = batch[['shop_id','item_id']]
                first_inner=False

            for i in range(FIRST_N_MONTH_TO_DROP):#exclude features where past can not be calculated
                
                if any(cr in path for cr in critical):
                    
                    batch=batch.drop(str(i), axis=1)
            
            rename_columns(batch, path)
            
            init = pd.merge(init, batch, on=['shop_id','item_id'], how = 'right')
            print(sum(batch.memory_usage()/10**6))
            #del batch
            gc.collect()

        if first:
            
            init.merge(item_shop_city_category_sup_category,on=['shop_id','item_id'], how='left').to_csv('data/merged.csv', index=False)
            first= False
        else:
            init.merge(item_shop_city_category_sup_category,on=['shop_id','item_id'], how='left').to_csv('data/merged.csv', mode='a', index=False, header=False)

    print('names in merged,',pathes )


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
                                         item_shop_city_category_sup_category=item_shop_city_category_sup_category,
                                         is_cnt=True)
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
                                         item_shop_city_category_sup_category=item_shop_city_category_sup_category,
                                         is_cnt=True)
        print(train)
        train.to_csv(f'data/value_{'_'.join(gr)}.csv', index=False)
        names.append(f'value_{'_'.join(gr)}')


    prices = create_item_shop_data(data_train, column='item_price')
    mean_prices_items = calculate_EMAs_pipeline(prices, #Worng
                                                ['item_id'], 
                                                item_shop_city_category_sup_category=item_shop_city_category_sup_category,
                                                alpha=1.0,
                                                is_cnt=False)
    changes = create_change(mean_prices_items)
    changes.to_csv('data/item_price_change.csv', index=False)
    names += ['item_price_change']
    return names


def calc_and_write_chunk(merged, data_train,item_shop_city_category_sup_category, chunksize_when_writing):

    alpha=1.0

    pathes = prepare_files(merged,data_train,item_shop_city_category_sup_category, alpha=alpha)

    shop_city=item_shop_city_category_sup_category[['shop_id','city']].drop_duplicates()
    category_super_category=item_shop_city_category_sup_category[['item_category_id','super_category']].drop_duplicates()

    merged=merged.merge(shop_city, on='shop_id', how='left')
    merged=merged.merge(category_super_category, on='item_category_id', how='left')

    #merged['city'] = merged['city'].map(lambda a:a[0])
    #merged['super_category'] = merged['super_category'].map(lambda a:a[0])

    prepare_for_boosting=True
    
    shop_item=item_shop_city_category_sup_category[['shop_id','item_id']].drop_duplicates()
    if prepare_for_boosting:
        merge_boosting(shop_item, pathes, chunksize_when_writing,item_shop_city_category_sup_category)

    
def create_split(past, chunk_size=300000):
    l = len(past)
    idxs = np.random.permutation(l)
    chunk_num = l // chunk_size if l%chunk_size==0  else   l // chunk_size+ 1
    i_l = []
    for i in range(chunk_num):
        i_l.append(idxs[i*chunk_size:(i+1)*chunk_size])

    return i_l

def prepare_batches(past, data_train,item_shop_city_category_sup_category, merged, chunk_size):

    
    
    l = len(past)
    idxs = create_split(past, chunk_size=chunk_size)
    for chunk in idxs:
        all_cartesians = past[chunk]
        all_cartesians=pd.DataFrame({'shop_id':all_cartesians[:,0], 'item_id':all_cartesians[:,1]})


        shop_city=item_shop_city_category_sup_category[['shop_id','city']].drop_duplicates()
        item_category=item_shop_city_category_sup_category[['item_id','item_category_id']].drop_duplicates()
        category_super_category=item_shop_city_category_sup_category[['item_category_id','super_category']].drop_duplicates()


        item_shop_city_category_sup_category_ret = all_cartesians.merge(shop_city).merge(item_category).merge(category_super_category)
        merged_ret = pd.merge(all_cartesians,merged, on=['item_id','shop_id'], how='left' ).fillna(0)

        yield merged_ret, item_shop_city_category_sup_category_ret

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
    item_category = whole.groupby('item_id')['item_category_id'].unique().reset_index()
    category_super_category = whole.groupby('item_category_id')['super_category'].unique().reset_index()

    item_shop_city_category_sup_category = merged.merge(shop_city). \
    merge(category_super_category)[['shop_id',	'item_id','item_category_id','city','super_category']]

    item_shop_city_category_sup_category['city'] = item_shop_city_category_sup_category['city'].map(lambda a:a[0])
    item_shop_city_category_sup_category['super_category'] = item_shop_city_category_sup_category['super_category'].map(lambda a:a[0])
    
    

    del whole
    del test

    past = prepare_past_ID_s_CARTESIAN(data_train)[-1][-1]

    first = True
    chunk_size=len(past)
    chunksize_when_writing = 200000

    for batch, item_shop_city_category_sup_category in prepare_batches(past,data_train,item_shop_city_category_sup_category, merged, chunk_size=chunk_size):
        calc_and_write_chunk(batch,data_train,item_shop_city_category_sup_category, chunksize_when_writing)
        first = False