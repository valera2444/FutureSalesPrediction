
from typing import Annotated, Literal

from fastapi import FastAPI, Query
from pydantic import BaseModel, Field
from sample_cascade_match import create_5_plus_match, download_s3_folder
import os, boto3
import pickle
import numpy as np
#import pandas as pd
from utils import select_columns_for_reading, prepare_data_validation_boosting, make_X_lag_format, append_some_columns
app = FastAPI()


def download_from_minio(args):

    minio_user=os.environ.get("MINIO_ACCESS_KEY")
    minio_password=os.environ.get("MINIO_SECRET_ACCESS_KEY")
    bucket_name = os.environ.get("BUCKET_NAME")
    
    ACCESS_KEY = minio_user
    SECRET_KEY =  minio_password
    host = 'http://localhost:9000'
    bucket_name = bucket_name

    s3c = boto3.resource('s3', 
                    aws_access_key_id=ACCESS_KEY,
                    aws_secret_access_key=SECRET_KEY,
                    endpoint_url=host) 
    
    download_s3_folder(s3c,bucket_name,args.path_data_cleaned, args.path_data_cleaned) # this must be used only once on server part 

    s3c = boto3.client('s3', 
                    aws_access_key_id=ACCESS_KEY,
                    aws_secret_access_key=SECRET_KEY,
                    endpoint_url=host)
    
    

    s3c.download_file(bucket_name, f'{args.path_for_merged}/merged.csv', f'{args.path_for_merged}/merged.csv') # ASSUMES THAT path args.path_for_merged exists +  this must be used only once on server part 
    
    s3c.download_file(bucket_name, f'{args.run_name}/lgbm.pkl', f'lgbm.pkl')

class Sample(BaseModel):
    item_id: int
    shop_id: int
    download:bool = False
    batch_size:int
    run_name: str

class Args:
    def __init__(self, d: dict):
        for el in d.keys():
            setattr(self, el, d[el])

@app.get("/predict/")
async def create_prediction(sample: Annotated[Sample, Query()]):

    args = Args({'run_name':sample.run_name,
            'path_for_merged':f'{sample.run_name}/data',
            'path_data_cleaned':f'{sample.run_name}/data/cleaned',
            'test_month':34,
            'item_id':sample.item_id,
            'shop_id':sample.shop_id,
            'batch_size':sample.batch_size})
    if sample.download:
        download_from_minio(args)
    
    model_filename='lgbm.pkl'
    with open(model_filename, "rb") as file:
        loaded_model = pickle.load(file)

    row = create_5_plus_match(args)

    row = prepare_data_validation_boosting(row, args.test_month)

    cols = select_columns_for_reading(row, args.test_month)

    X_val=row[cols]
    X_val = X_val.drop('item_id', axis=1)
    X_val['shop_id'] = X_val['shop_id'].astype('category')
    X_val['item_category_id'] = X_val['item_category_id'].astype('category')
    X_val['city'] = X_val['city'].astype('category')
    X_val['super_category'] = X_val['super_category'].astype('category')

    X_val = make_X_lag_format(X_val, args.test_month)

    X_val = append_some_columns(X_val, args.test_month )

    print(loaded_model.feature_names_in_)
    print(X_val.columns)
    prediction = loaded_model.predict(X_val, validate_features=True)

    print(np.clip(prediction[0],0,20))
    return f"Prediction for shop {args.shop_id} item {args.item_id} is {np.clip(prediction[0],0,20)}"