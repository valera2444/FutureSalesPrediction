from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.decorators import task, dag
from airflow.models.baseoperator import chain

import datetime as dt

p = {'trials':2, 
     'run_name':'Enter path where your source data folder stored in s3 storage.',

     'lr_range_optuna':"0.07 0.1",
     'max_num_leaves_range_optuna':"32 64",
     'num_estimators_range_optuna':"100 100",

     'batch_size_for_train':30000,
     'test_month':34,
     'batches_for_train':1}



@dag(dag_id="predict", start_date=dt.datetime(2023, 11, 16),schedule_interval=None, params=p)
def run_pipeline():
    
    run_name = "{{params['run_name']}} "
    data_path = "{{params['run_name']}}/data "
    data_cleaned_path = "{{params['run_name']}}/data/cleaned "
    images_storage_path = "{{params['run_name']}}/images "
    lr_range_optuna = "{{params['lr_range_optuna']}} "
    max_num_leaves_range_optuna = "{{params['max_num_leaves_range_optuna']}} "
    num_estimators_range_optuna = "{{params['num_estimators_range_optuna']}} "
    batch_size_for_train = "{{params['batch_size_for_train']}} "
    test_month = "{{params['test_month']}} "
    batches_for_train= "{{params['batches_for_train']}} "

    t1 = BashOperator(
        task_id="etl",
        bash_command="python3.12 ${AIRFLOW_HOME}/future_sales_prediction/etl.py "  + f"--run_name {run_name} --source_path {data_path} --destination_path {data_cleaned_path}"
    )

    t2 = BashOperator(
        task_id="prepare_data",
        bash_command="python3.12 ${AIRFLOW_HOME}/future_sales_prediction/prepare_data.py "+f"--run_name {run_name} --source_path  {data_cleaned_path} --destination_path  {data_path} --test_month {test_month}"
    )

    t3 = BashOperator(
        task_id="hyperparameters_tuning",
        bash_command="python3.12 ${AIRFLOW_HOME}/future_sales_prediction/hyperparameters_tuning.py "+ \
            f"--run_name {run_name} --path_for_merged  {data_path} --path_data_cleaned   {data_cleaned_path}"+ \
            "--n_trials  {{params['trials']}} " +\
            f"--max_num_leaves_range_optuna {max_num_leaves_range_optuna} " + \
            f"--num_estimators_range_optuna {num_estimators_range_optuna} " + \
            f"--lr_range_optuna {lr_range_optuna} " + \
            f"--batch_size_for_train {batch_size_for_train} " +\
            f"--test_month {test_month}"
    )

    t4 = BashOperator(
        task_id="validate_12_monthes",
        bash_command="python3.12 ${AIRFLOW_HOME}/future_sales_prediction/create_submission.py " + \
            f"--run_name {run_name} --path_for_merged {data_path} --path_data_cleaned  {data_cleaned_path} --is_create_submission 0 --batch_size_for_train {batch_size_for_train} --test_month {test_month} --batches_for_train {batches_for_train}"
    )
    t5 = BashOperator(
        task_id="create_submission",
        bash_command="python3.12 ${AIRFLOW_HOME}/future_sales_prediction/create_submission.py " + \
            f"--run_name {run_name} --path_for_merged {data_path} --path_data_cleaned  {data_cleaned_path} --is_create_submission 1 --batch_size_for_train {batch_size_for_train} --test_month {test_month} --batches_for_train {batches_for_train}"
    )

    t6 = BashOperator(
        task_id="error_analysis",
        bash_command="python3.12 ${AIRFLOW_HOME}/future_sales_prediction/error_analysis.py " +  \
            f"--run_name {run_name} --path_for_merged {data_path} --path_data_cleaned {data_cleaned_path} --path_artifact_storage {images_storage_path} --test_month {test_month}"
    )


    t7 = BashOperator(
        task_id="model_interpret",
        bash_command="python3.12 ${AIRFLOW_HOME}/future_sales_prediction/model_interpret.py " + \
            f"--run_name {run_name} --path_for_merged {data_path} --path_data_cleaned {data_cleaned_path} --path_artifact_storage {images_storage_path} --batch_size_for_train {batch_size_for_train} --test_month {test_month}"
    )

    chain(t1, t2, t3, t4, t5, t6, t7)

run_pipeline()