import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


if __name__=='__main__':
    data_train = pd.read_csv('data/sales_train.csv')
    data_test = pd.read_csv('data/test.csv')

    item_cat = pd.read_csv('data/item_categories.csv')
    items = pd.read_csv('data/items.csv')
    shops = pd.read_csv('data/shops.csv')
    shops=shops.drop([0,1,10])
    data_train.loc[data_train['shop_id']==0,'shop_id'] = 57
    data_train.loc[data_train['shop_id']==1,'shop_id'] = 58
    data_train.loc[data_train['shop_id']==10,'shop_id'] = 11

    data_test.loc[data_test['shop_id']==0,'shop_id'] = 57
    data_test.loc[data_test['shop_id']==1,'shop_id'] = 58
    data_test.loc[data_test['shop_id']==10,'shop_id'] = 11

    data_train = data_train.drop(data_train[data_train['item_price'] < 0].index)
    data_train = data_train.drop(data_train[data_train['item_price'] > 100000].index)

    data_train = data_train.drop(data_train[data_train['item_cnt_day'] < 0].index)

    data_train=data_train.drop_duplicates()

    data_train.to_csv('data_cleaned/data_train.csv', mode='w',index=False)
    shops.to_csv('data_cleaned/shops.csv', mode='w',index=False)
    item_cat.to_csv('data_cleaned/item_categories.csv', mode='w',index=False)
    items.to_csv('data_cleaned/items.csv', mode='w',index=False)
    data_test.to_csv('data_cleaned/test.csv', mode='w',index=False)
