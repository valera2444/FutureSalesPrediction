import pandas as pd
import numpy as np
from pathlib import Path
import click
import argparse
import mlflow
import os
import boto3
from io import StringIO
from io import BytesIO

def run_etl(client, source_path, destination_path):
    """
    runs etl and writes cleaned data into data_cleaned/{file_name}.csv

    Args:
        source_path (str): path to the source data folder
        destination_path (str): path to the destination data folder
    """
    #data_train = pd.read_csv(f'{source_path}/sales_train.csv')
    #data_test = pd.read_csv(f'{source_path}/test.csv')

    #item_cat = pd.read_csv(f'{source_path}/item_categories.csv')
    #items = pd.read_csv(f'{source_path}/items.csv')
    #shops = pd.read_csv(f'{source_path}/shops.csv')
    

    bucket_name = 'mlflow'
    

    object_key = f'{source_path}/sales_train.csv'
    csv_obj = client.get_object(Bucket=bucket_name, Key=object_key)
    body = csv_obj['Body']
    csv_string = body.read().decode('utf-8')
    data_train = pd.read_csv(StringIO(csv_string))

    object_key = f'{source_path}/item_categories.csv'
    csv_obj = client.get_object(Bucket=bucket_name, Key=object_key)
    body = csv_obj['Body']
    csv_string = body.read().decode('utf-8')
    item_cat = pd.read_csv(StringIO(csv_string))

    object_key = f'{source_path}/test.csv'
    csv_obj = client.get_object(Bucket=bucket_name, Key=object_key)
    body = csv_obj['Body']
    csv_string = body.read().decode('utf-8')
    data_test = pd.read_csv(StringIO(csv_string))

    object_key = f'{source_path}/shops.csv'
    csv_obj = client.get_object(Bucket=bucket_name, Key=object_key)
    body = csv_obj['Body']
    csv_string = body.read().decode('utf-8')
    shops = pd.read_csv(StringIO(csv_string))

    object_key = f'{source_path}/items.csv'
    csv_obj = client.get_object(Bucket=bucket_name, Key=object_key)
    body = csv_obj['Body']
    csv_string = body.read().decode('utf-8')
    items = pd.read_csv(StringIO(csv_string))

    
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

    #data_train.to_csv(f'{destination_path}/data_train.csv', mode='w',index=False)
    #shops.to_csv(f'{destination_path}/shops.csv', mode='w',index=False)
    #item_cat.to_csv(f'{destination_path}/item_categories.csv', mode='w',index=False)
    #items.to_csv(f'{destination_path}/items.csv', mode='w',index=False)
    #data_test.to_csv(f'{destination_path}/test.csv', mode='w',index=False)

    csv_bytes = data_train.to_csv(index=False).encode('utf-8')
    csv_buffer = BytesIO(csv_bytes)
    client.put_object(Body=csv_buffer,
                    Bucket='mlflow',
                    Key=f'{destination_path}/data_train.csv')
    
    csv_bytes = item_cat.to_csv(index=False).encode('utf-8')
    csv_buffer = BytesIO(csv_bytes)
    client.put_object(Body=csv_buffer,
                    Bucket='mlflow',
                    Key=f'{destination_path}/item_categories.csv')
    
    csv_bytes = shops.to_csv(index=False).encode('utf-8')
    csv_buffer = BytesIO(csv_bytes)
    client.put_object(Body=csv_buffer,
                    Bucket='mlflow',
                    Key=f'{destination_path}/shops.csv')
    
    csv_bytes = items.to_csv(index=False).encode('utf-8')
    csv_buffer = BytesIO(csv_bytes)
    client.put_object(Body=csv_buffer,
                    Bucket='mlflow',
                    Key=f'{destination_path}/items.csv')
    
    csv_bytes = data_test.to_csv(index=False).encode('utf-8')
    csv_buffer = BytesIO(csv_bytes)
    client.put_object(Body=csv_buffer,
                    Bucket='mlflow',
                    Key=f'{destination_path}/test.csv')
    




def get_or_create_experiment(experiment_name):
    """
    Retrieve the ID of an existing MLflow experiment or create a new one if it doesn't exist.

    This function checks if an experiment with the given name exists within MLflow.
    If it does, the function returns its ID. If not, it creates a new experiment
    with the provided name and returns its ID


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

    
    
    #If omit this line, running wuth docker gives error. If using MLFlow Projects, git is required?
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_path', type=str, help='folder where source data stored in minio')
    parser.add_argument('--destination_path', type=str, help='folder where data after etl will be stored')

    args = parser.parse_args()

    client = boto3.client('s3',
                      endpoint_url='http://minio:9000',
                      aws_access_key_id='airflow_user',
                      aws_secret_access_key='airflow_paswword')


    run_etl(client, args.source_path, args.destination_path)