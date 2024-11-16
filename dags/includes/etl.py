import pandas as pd
import numpy as np
from pathlib import Path
import click
@click.command()
@click.option('--source_path')
@click.option('--destination_path')
def run_etl(source_path, destination_path):
    """
    runs etl and writes cleaned data into data_cleaned/{file_name}.csv

    Args:
        source_path (str): path to the source data folder
        destination_path (str): path to the destination data folder
    """
    data_train = pd.read_csv(f'{source_path}/sales_train.csv')
    data_test = pd.read_csv(f'{source_path}/test.csv')

    item_cat = pd.read_csv(f'{source_path}/item_categories.csv')
    items = pd.read_csv(f'{source_path}/items.csv')
    shops = pd.read_csv(f'{source_path}/shops.csv')
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

    
    Path(destination_path).mkdir(parents=True, exist_ok=True)

    data_train.to_csv(f'{destination_path}/data_train.csv', mode='w',index=False)
    shops.to_csv(f'{destination_path}/shops.csv', mode='w',index=False)
    item_cat.to_csv(f'{destination_path}/item_categories.csv', mode='w',index=False)
    items.to_csv(f'{destination_path}/items.csv', mode='w',index=False)
    data_test.to_csv(f'{destination_path}/test.csv', mode='w',index=False)
