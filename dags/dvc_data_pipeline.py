from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import timedelta, datetime
import os

def pull_data_from_dvc():
    os.system('dvc pull')

def process_data():
    print("Processing the data...")

def push_data_to_dvc():
    os.system('dvc push')

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2025, 5, 9),
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'dvc_data_pipeline',
    default_args=default_args,
    description='A simple DVC data pipeline with Airflow',
    schedule_interval="*/30 * * * *",  # ⏱️ Every 30 minutes
    catchup=False,
) as dag:

    pull_task = PythonOperator(
        task_id='pull_data_from_dvc',
        python_callable=pull_data_from_dvc,
    )

    process_task = PythonOperator(
        task_id='process_data',
        python_callable=process_data,
    )

    push_task = PythonOperator(
        task_id='push_data_to_dvc',
        python_callable=push_data_to_dvc,
    )

    pull_task >> process_task >> push_task
