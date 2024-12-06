# Fututre sales prediction

## Introduction

Future sales prediction for the cartesian product of known shops and items for the next month. Project provides opportunity to run full pipeline from raw data to the final predictions.


## Start servers
Steps for running all servers locally:
1) Create .env file with variables same as in env_example.txt
3) Run `docker compose up airflow-init`
4) Run `docker compose up`
5) Create folder with the name you want in s3 bucket from your .enf file
6) Create folder with the name "data" inside the folder you have just created and load raw .csv files from kaggle in data folder

Once you have performed all this actions, the next servers will start up:
1) airflow (locallhost:8080)
2) mlflow (locallhost:5000)
3) minio (locallhost:9001)
4) fastapi (locallhost:5050)

## Airflow
You can run your dag manually any time you want after all necessary data have been loaded. When running DAG, you must specify several parameters:
1) `trials` - number of iterations for optuna hyperparameter optimization.
2) `run_name` - pipeline identifier, must be the same as you have chosen in 5) from start servers section
3) `lr_range_optuna` - learning rate range for optuna
4) `max_num_leaves_range_optuna` - maximum number of leaves in one tree for optuna
5) `num_estimators_range_optuna` - number of trees for optuna
6) `batch_size_for_train` - batch size for training
7) `test_month` - date block number of month from test.csv
8) `batches_for_train` - number of bacthes to train on during submission creation and validation on 12 monthes. Not used during hyperparameter optimization

Preferred parameters for systems with RAM capacity >= 10GB are:. 
1) lr_range_optuna: 0.0001 0.1
2) max_num_leaves_range_optuna: 32 256
3) num_estimators_range_optuna: 100 600
4) batch_size_for_train: 3_000_000
5) batches_for_train: 3
6) trials: this hyperparameter influence on hyperparameter execution time.
 > **Note**: Whole pipilene execution time strongly depends on number of avaliable cores as lightgbm and batch preparation use multiprocessing. 

## MLFlow
This service is responsible for experiment tracking. You may see hyperparameter optimization processes details under hyperparameter_optimization experiment and run name that you have chosen previously.

You can also find results of 12 month validation here under create_submission experiment.

## Minio
This service is a data storage which is used both as backup storage and artifact storage. 

Root path for each pipeline run is s3://bucket_name/run_name. After pipeline execution, there will be few folders and files:
1) `best_parameters.pkl`: serialized dictionary with best hyperparamers after running optuna.
2) `data`: folder, that contains all data used during pipeline execution.
3) `images`: folder, that contains plots of model interpretability and error analysis sections.
4) `submission.csv`: predictions for test month.

## FastAPI
Endpoint for model online inference after all airflow job done.

This service reads data after etl, merged.csv and lgbm.pkl from minio after airflow prepared it.


Have 2 path operatons:

1) `download_data`: Enables to download data for specific run to the server
2) `predict`: Function for online inference.
Returns predicted sales in [0,20] range for passed item and shop id.
If data after etl doesnt contain one of them, application exists with error.
Batch size used not to read whole merged.csv into RAM because memory error posiible. The less batch size - the more inference time

More info about this service can be found on http://localhost:5050/docs