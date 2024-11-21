from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.decorators import task, dag
from airflow.models.baseoperator import chain

import datetime as dt

p = {'trials':2, 'run_name':'Enter path where your source data folder stored in s3 storage.'}



@dag(dag_id="predict", start_date=dt.datetime(2023, 11, 16),schedule_interval=None, params=p)
def run_pipeline():
    #{{params['run_name']}}/
    data_path = "data "
    data_cleaned_path = "data/cleaned "
    images_storage_path = "data/images "

    t1 = BashOperator(
        task_id="etl",
        bash_command="python3.12 ${AIRFLOW_HOME}/future_sales_prediction/etl.py"  + f"--source_path {data_path} --destination_path {data_cleaned_path}"
    )

    t2 = BashOperator(
        task_id="prepare_data",
        bash_command="python3.12 ${AIRFLOW_HOME}/future_sales_prediction/prepare_data.py"+ f"--source_path  {data_cleaned_path} --destination_path  {data_path} "
    )

    t3 = BashOperator(
        task_id="hyperparameters_tuning",
        bash_command="python3.12 ${AIRFLOW_HOME}/future_sales_prediction/hyperparameters_tuning.py"+ f"--path_for_merged  {data_path} --path_data_cleaned   {data_cleaned_path}"+ "--n_trials  {{params['trials']}} "
    )

    t4 = BashOperator(
        task_id="validate_12_monthes",
        bash_command="python3.12 ${AIRFLOW_HOME}/future_sales_prediction/create_submission.py" + f"--path_for_merged {data_path} --path_data_cleaned  {data_cleaned_path} --is_create_submission 0 "
    )
    t5 = BashOperator(
        task_id="create_submission",
        bash_command="python3.12 ${AIRFLOW_HOME}/future_sales_prediction/create_submission.py" + f"--path_for_merged {data_path} --path_data_cleaned  {data_cleaned_path} --is_create_submission 1 "
    )

    t6 = BashOperator(
        task_id="error_analysis",
        bash_command="python3.12 ${AIRFLOW_HOME}/future_sales_prediction/error_analysis.py" +  f"--path_for_merged {data_path} --path_data_cleaned {data_cleaned_path} --path_artifact_storage {images_storage_path} "
    )


    t7 = BashOperator(
        task_id="model_interpret",
        bash_command="python3.12 ${AIRFLOW_HOME}/future_sales_prediction/model_interpret.py" + f"--path_for_merged {data_path} --path_data_cleaned {data_cleaned_path} --path_artifact_storage {images_storage_path} "
    )

    chain(t1, t2,t3, t4,t5,t6, t7)

run_pipeline()