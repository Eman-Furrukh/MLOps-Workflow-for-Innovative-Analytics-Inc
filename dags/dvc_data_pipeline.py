from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import timedelta, datetime
import os

def collect_data():
    os.system('python src/collect_data.py')

def update_dvc():
    # DO NOT use `dvc add` again and again!
    os.system('git add data/raw/glasgow_weather_data.csv')
    os.system('git commit -m "Append weather data from Airflow"')
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
    description='A DVC-integrated pipeline that collects and pushes data every 10 minutes',
    schedule_interval="*/10 * * * *",  # ⏱️ Every 10 minutes
    catchup=False,
) as dag:

    collect_data_task = PythonOperator(
        task_id='collect_data',
        python_callable=collect_data,
    )

    update_dvc_task = PythonOperator(
        task_id='update_dvc',
        python_callable=update_dvc,
    )

    collect_data_task >> update_dvc_task
