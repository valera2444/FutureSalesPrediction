from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.decorators import task, dag
from airflow.utils.dates import days_ago
from airflow.models.baseoperator import chain
from airflow.utils.dates import days_ago
import datetime as dt

p = {'trials':2}



@dag(dag_id="predict", start_date=dt.datetime(2023, 11, 16),schedule_interval=None, params=p)
def run_pipeline():

    t1 = BashOperator(
        task_id="etl",
        bash_command="python3.12 ${AIRFLOW_HOME}/future_sales_prediction/etl.py --source_path data --destination_path data/cleaned "
    )

    t2 = BashOperator(
        task_id="prepare_data",
        bash_command="python3.12 ${AIRFLOW_HOME}/future_sales_prediction/prepare_data.py --source_path  data/cleaned --destination_path  data "
    )

    t3 = BashOperator(
        task_id="hyperparameters_tuning",
        bash_command="python3.12 ${AIRFLOW_HOME}/future_sales_prediction/hyperparameters_tuning.py --path_for_merged  data --path_data_cleaned   data/cleaned --n_trials  {{params['trials']}} "
    )

    t4 = BashOperator(
        task_id="create_submission",
        bash_command="python3.12 ${AIRFLOW_HOME}/future_sales_prediction/create_submission.py --path_for_merged data --path_data_cleaned   data/cleaned/cleaned "
    )

    chain(t1, t2,t3, t4)

run_pipeline()