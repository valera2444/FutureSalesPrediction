import pandas as pd
import numpy as np
import sklearn
from sklearn.metrics import root_mean_squared_error
from collections import defaultdict


from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression

import multiprocessing

import time

import pickle

import shap

import os

import argparse

import mlflow

import matplotlib.pyplot as plt

from data_preprocess_utils import create_batch_train, create_batch_val, make_X_lag_format, append_some_columns, prepare_past_ID_s_CARTESIAN

from gcloud_operations import upload_folder, upload_file, download_file, download_folder

def train_model_and_create_shaps(model, batch_size, val_month, shop_item_pairs_WITH_PREV_in_dbn,batch_size_to_read,batches_for_training,shop_item_pairs_in_dbn,source_path):
    """
    Trains a machine learning model with specified batches and tracks RMSE for training on current val_month.
    Args:
        model (object): Machine learning model to be trained.
        batch_size (int): Number of samples per batch.
        val_month (int): Month used for validation.
        shop_item_pairs_WITH_PREV_in_dbn (np.array[np.array[np.array[int,int]]]):  Accumulated cartesian products for each time block.
        batch_size_to_read (int): chunck size when reading csv file. This prevents memory error when using multiple processes
        batches_for_training (int): number of batches to train on. These parameter may be extremelly usefull for models which doesnt support batch learning. Also this may be usefull for reducing train time
        shop_item_pairs_in_dbn (pd.DataFrame, optional): dataframe of cartesian products of (shop, item) for date_block_nums. If None, validation is not performed after every batch

    Returns:
        tuple: 
            - model (object): Trained model.
            - columns_order (list[str]): Ordered list of feature columns.
    """
    first=True
    rmse = 0
    c=0
    columns_order=None
    
    Y_true_l = []
    preds_l = []
    for X_train,Y_train  in create_batch_train(batch_size, val_month,shop_item_pairs_WITH_PREV_in_dbn,batch_size_to_read,source_path):
        #print(f'train on batch {c} started')
        t1_batch = time.time()
        t1_data_prep = time.time()
        #print(f'data preparation on batch {c} started')
        if X_train is None:
            print('None')
            continue

        if type(model) in [Lasso,SVC]:
            #print(X_train.columns)
            X_train.drop('shop_id', inplace=True, axis=1) 
            X_train.drop('item_category_id', inplace=True, axis=1) 
            X_train.drop('item_id', inplace=True, axis=1)
            X_train.drop('city', inplace=True, axis=1)
            X_train.drop('shop_id', inplace=True, axis=1)
        elif type(model) ==LGBMRegressor:
            #print(list(X_train.columns))
        
            X_train = X_train.drop('item_id', axis=1)
            X_train['shop_id'] = X_train['shop_id'].astype('category')
            X_train['item_category_id'] = X_train['item_category_id'].astype('category')
            X_train['city'] = X_train['city'].astype('category')
            X_train['super_category'] = X_train['super_category'].astype('category')
            
            pass
        
        
            
        Y_train = np.clip(Y_train,0,20)
        
        if X_train.empty:
            print('None')
            continue
    
        
        columns_order=X_train.columns

        t2_data_prep = time.time()
        #print(f'data preparation on batch {c} time:',t2_data_prep-t1_data_prep)
        #print('model fitting started')
        t1_fit = time.time()
        if c == 0:
            pass
            #print('train columns')
            #print(X_train.columns)
        if type(model) in [Lasso,SVC]:
            model.fit(X_train, Y_train)
            y_train_pred = model.predict(X_train)
        
        elif type(model) == LGBMRegressor:
            if first:
                model.fit(X_train, Y_train)
                first=False
            else:
                model.fit(X_train, Y_train, init_model=model)
            y_train_pred = model.predict(X_train, validate_features=True)


        elif type(model) == RandomForestRegressor:
            model.fit(X_train, Y_train)
            y_train_pred = model.predict(X_train)  
        t2_fit = time.time()
        #print(f'model fitting time on batch {c},',t2_fit - t1_fit)
        
        Y_true_l.append(Y_train)
        preds_l.append(y_train_pred)
        t2_batch = time.time()
        print(f'train on batch {c} time,',t2_batch-t1_batch)
        
        
            

        c+=1

        if c == batches_for_training:
            break
    
    if shop_item_pairs_in_dbn is not None:
        model,explainer,shap_values,X_val_to_ret = create_shaps(model,batch_size, val_month,columns_order, shop_item_pairs_in_dbn,batch_size_to_read,source_path)
        
        

    return model,explainer,shap_values,X_val_to_ret

def create_shaps(model,batch_size, val_month, columns_order, shop_item_pairs_in_dbn, batch_size_to_read,source_path):
    """
    Creates shapes 

    Args:
        model (object): Machine learning model to be validated.
        batch_size (int): Number of samples per batch. MUST BE > 400_000
        val_month (int): Month used for validation.
        columns_order (list[str]): Order of feature columns.
        shop_item_pairs_in_dbn (pd.DataFrame): dataframe of cartesian products of (shop, item) for date_block_nums
        batch_size_to_read (int): chunck size when reading csv file. This prevents memory error when using multiple processes

    Returns:
        tuple: 
            - model
            - explainer
            - shap_values
            - X_val_to_ret
    """
    rmse = 0
    c=0
    
    val_preds = []
    Y_true_l = []
    preds_l = []
    Y_true_l_shop_item=[]
    
    for X_val, Y_val in create_batch_val(val_month, shop_item_pairs_in_dbn, batch_size_to_read,source_path): #but then cartesian product used
        shop_id = X_val.shop_id
        item_id = X_val.item_id
        
        
        if type(model) in [sklearn.linear_model._coordinate_descent.Lasso,
                            SVC]:
            
            X_val.drop('shop_id', inplace=True, axis=1) 
            X_val.drop('item_category_id', inplace=True, axis=1) 
            X_val.drop('item_id', inplace=True, axis=1)
            X_val.drop('city', inplace=True, axis=1)
            X_val.drop('shop_id', inplace=True, axis=1)
            

        elif type(model) ==LGBMRegressor:
            
            X_val = X_val.drop('item_id', axis=1)
            X_val['shop_id'] = X_val['shop_id'].astype('category')
            X_val['item_category_id'] = X_val['item_category_id'].astype('category')
            X_val['city'] = X_val['city'].astype('category')
            X_val['super_category'] = X_val['super_category'].astype('category')
                    
            pass
            
        
            
        Y_val = np.clip(Y_val,0,20)
        
        
        X_val = make_X_lag_format(X_val, val_month)
        
        X_val=append_some_columns(X_val, val_month)
        X_val = X_val[columns_order]

        print('shap calculation started...')
        
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_val)
        X_val_to_ret=X_val
    


    return model,explainer,shap_values,X_val_to_ret

def validate_ML(params,batch_size,val_monthes, shop_item_pairs_in_dbn, shop_item_pairs_WITH_PREV_in_dbn,batch_size_to_read, batches_for_training, source_path):
    """
    Runs model training and validation across multiple months and computes RMSE for each month.
    Note: batch learning works properly only for LGBMRegressor. Using another model with batch_size<=len(shop_item_pairs_WITH_PREV_in_dbn[dbn]) will lead to incorrect results
    Args:
        params (dict): params for machine learning model.
        batch_size (int): Number of samples per batch.
        val_monthes (list): List of months for validation.
        shop_item_pairs_in_dbn (pd.DataFrame): dataframe of cartesian products of (shop, item) for date_block_nums
        shop_item_pairs_WITH_PREV_in_dbn ( np.array[np.array[np.array[int,int]]] ): array of accumulated cartesian products of (shop, item) for date_block_nums
        batch_size_to_read (int): chunck size when reading csv file. This prevents memory error when using multiple processes
        batches_for_training (int): number of batches to train on. These parameter may be extremelly usefull for models which doesnt support batch learning. Also this may be usefull for reducing train time
    Returns:
        tuple: 
            - model
            - explainer
            - shap_values
            - X_val_to_ret
    """
    
    
    for val_month in val_monthes:

        run_name=f'validation on {val_month}'

        model = LGBMRegressor(
            num_leaves=params['num_leaves'],
            n_estimators=params['n_estimators'],
            learning_rate=params['learning_rate'],
            verbose=-1,
            n_jobs=multiprocessing.cpu_count()
        )
        print(f'month {val_month} started')
        t1 = time.time()
        
        print('month', val_month%12)


        model,explainer,shap_values,X_val_to_ret = train_model_and_create_shaps(
            model, 
            batch_size, 
            val_month,
            shop_item_pairs_WITH_PREV_in_dbn,
            batch_size_to_read,
            batches_for_training,
            shop_item_pairs_in_dbn = shop_item_pairs_in_dbn,
            source_path=source_path
        )

        
        

    return model,explainer,shap_values,X_val_to_ret


def download_s3_folder(s3c, bucket_name, s3_folder, local_dir=None):
    """
    Download the contents of a folder directory to local_dir (creates if not exist)
    Args:
        s3c: s3 resource
        bucket_name: the name of the s3 bucket
        s3_folder: the folder path in the s3 bucket
        local_dir: a relative or absolute directory path in the local file system
    """
    bucket = s3c.Bucket(bucket_name)
    for obj in bucket.objects.filter(Prefix=s3_folder):
        target = obj.key if local_dir is None \
            else os.path.join(local_dir, os.path.relpath(obj.key, s3_folder))
        if not os.path.exists(os.path.dirname(target)):
            os.makedirs(os.path.dirname(target))
        if obj.key[-1] == '/':
            continue
        bucket.download_file(obj.key, target)
        #print(obj.key, target)


def get_or_create_experiment(experiment_name):
    """
    Retrieve the ID of an existing MLflow experiment or create a new one if it doesn't exist.

    This function checks if an experiment with the given name exists within MLflow.
    If it does, the function returns its ID. If not, it creates a new experiment
    with the provided name and returns its ID.

    Parameters:
    - experiment_name (str): Name of the MLflow experiment.

    Returns:
    - str: ID of the existing or newly created MLflow experiment.
    """

    if experiment := mlflow.get_experiment_by_name(experiment_name):
        return experiment.experiment_id
    else:
        return mlflow.create_experiment(experiment_name)
    
def run_create_submission(path_for_merged, path_data_cleaned,batch_size_for_train, test_month):
    """
    Function for expanding window validation or creating submission. Reads hyperparameters from saved_dictionary.pkl and writes submission in the root

    Args:
        path_for_merged (str): path to the folder where created by prepare_data.py file stored
        path_data_cleaned (str): path to the folder where created by etl.py file stored
        is_create_submission (bool): whether to infer model with sliding winfow validation or create submission
        batch_size_for_train(int): 
        test_month (int): 
    Returns:
        explainer,shap_values,X_val_to_ret
    """

    


    data_train = pd.read_csv(f'{path_data_cleaned}/data_train.csv')

    shop_item_pairs_in_dbn, shop_item_pairs_WITH_PREV_in_dbn = prepare_past_ID_s_CARTESIAN(data_train)

    #parametrs which doesnt affect training
    batch_size_to_read=50_000 # the more batches_for_training - the less should be batch_size_to_read to prevent memory error

    #parametrs which affect number of estimators:

    #using batches_for_training > 1 doesnt improve metrics much

    batches_for_training=1

    batch_size=batch_size_for_train
    

    with open(f'{args.run_name}/best_parameters.pkl', 'rb') as f:
        params = pickle.load(f)

    #model parameters
    num_leaves=params['num_leaves']
    n_estimators=params['n_estimators']
    learning_rate=params['lr']

    create_submission=False
    
    if not create_submission:

        
        val_monthes=[test_month-1]

        print('validation started...')
        

        params = {'num_leaves':num_leaves,
                    'n_estimators':n_estimators,
                    'learning_rate':learning_rate,
                    'batch_size_to_read':batch_size_to_read,
                    'batches_for_training':batches_for_training,
                    'batch_size':batch_size
                    }
        model,explainer,shap_values,X_val_to_ret = validate_ML(
                                            params,
                                            batch_size=batch_size,
                                            val_monthes=val_monthes, 
                                            shop_item_pairs_in_dbn=shop_item_pairs_in_dbn,
                                            shop_item_pairs_WITH_PREV_in_dbn=shop_item_pairs_WITH_PREV_in_dbn,
                                            batch_size_to_read=batch_size_to_read,
                                            batches_for_training=batches_for_training,
                                            source_path=path_for_merged
        )
        
        
        return explainer,shap_values,X_val_to_ret




if __name__ == '__main__':


    parser = argparse.ArgumentParser()

    parser.add_argument('--test_month', type=int, help='month to create submission on')

    parser.add_argument('--run_name', type=str)
    parser.add_argument('--path_for_merged', type=str,help='folder where merged.csv stored after prepare_data.py. Also best hyperparameters  stored in this folder')
    parser.add_argument('--path_data_cleaned', type=str, help='folder where data stored after etl')
    parser.add_argument('--path_artifact_storage', type=str, help='folder where shap images will be stored')
    parser.add_argument('--batch_size_for_train', type=int)
    

    args = parser.parse_args()

    bucket_name = os.environ.get("BUCKET_NAME")


    if not os.path.exists(args.path_data_cleaned):
        download_folder(bucket_name,args.path_data_cleaned)

    if not os.path.exists(f'{args.path_for_merged}/merged.csv'):
        download_file(bucket_name, f'{args.path_for_merged}/merged.csv')

    if not os.path.exists(f'{args.run_name}/best_parameters.pkl'):
        download_file(bucket_name, f'{args.run_name}/best_parameters.pkl')
    
    mlflow.set_tracking_uri(uri="http://mlflow:5000")

    explainer, shaps, X_display = run_create_submission(args.path_for_merged, args.path_data_cleaned,args.batch_size_for_train, args.test_month)

    run_id = mlflow.search_runs(experiment_names=['create_submission'],filter_string=f"run_name='{args.run_name}'").iloc[0]['run_id']

    shap.plots.beeswarm(shaps, max_display=20,show=False)
    plt.title('SHAPs')
    plt.savefig('shaps_beeswarm.png',bbox_inches='tight')
    plt.close()
    print('Run id to save plots:', run_id)
    mlflow.log_artifact(local_path='shaps_beeswarm.png', artifact_path= f'{args.path_artifact_storage}/shaps', run_id=run_id) 
    
    shap.plots.bar(shaps * 100, max_display=20,show=False)
    plt.savefig('shaps_bar.png',bbox_inches='tight')
    mlflow.log_artifact(local_path='shaps_bar.png', artifact_path= f'{args.path_artifact_storage}/shaps', run_id=run_id) 

    plt.close()
    