
from typing import Annotated, Literal

from fastapi import FastAPI, Query
from pydantic import BaseModel, Field
from create_row_in_req_format import create_5_plus_match
import os, boto3
import pickle
import numpy as np
#import pandas as pd

from utils import select_columns_for_reading, prepare_data_validation_boosting, make_X_lag_format, append_some_columns

from download_data import download_from_minio

from fastapi import FastAPI

description = """
Endpoint for model online inference after all airflow job done. 

## download_data

This enables to download data for specific run to the server

## predict

Function for online inference.
Returns predicted sales in [0,20] range for passed item and shop id.
If data after etl doesnt contain one of them, application exists with error.
Batch size used not to read whole merged.csv into RAM because memory error posiible. The less batch size - the more inference time
"""

tags_metadata = [
    {
        "name": "predict",
        "description": "Makes prediction on shop, item pair",
    },
    {
        "name": "download_data",
        "description": "Downloads data from specified run_name to server",
        
    }
]


app = FastAPI(
    title="Predict-future-sales",
    description=description,
    summary="Online inference endpoint",
    openapi_tags=tags_metadata
)




class SampleParams(BaseModel):
    item_id: int
    shop_id: int
    batch_size:int
    run_name: str

class DataDownloadParams(BaseModel):
    run_name: str

class Args:
    def __init__(self, d: dict):
        for el in d.keys():
            setattr(self, el, d[el])

def prepare_row(args):

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
    
    return X_val

@app.get('/download_data/', tags=["download_data"])
async def download_data(params: Annotated[DataDownloadParams, Query()]):
    args = Args({'run_name':params.run_name,
            'path_for_merged':f'{params.run_name}/data',
            'path_data_cleaned':f'{params.run_name}/data/cleaned'})
    
    download_from_minio(args)

    return "Data downloaded successfully"


@app.get("/predict/", tags=["predict"])
async def create_prediction(sample: Annotated[SampleParams, Query()]):

    args = Args({'run_name':sample.run_name,
            'path_for_merged':f'{sample.run_name}/data',
            'path_data_cleaned':f'{sample.run_name}/data/cleaned',
            'test_month':34,
            'item_id':sample.item_id,
            'shop_id':sample.shop_id,
            'batch_size':sample.batch_size})
    
    
    model_filename='lgbm.pkl'

    with open(model_filename, "rb") as file:
        loaded_model = pickle.load(file)

    row = prepare_row(args)

    prediction = loaded_model.predict(row, validate_features=True)


    return f"Prediction for shop {args.shop_id} item {args.item_id} is {np.clip(prediction[0],0,20)}"