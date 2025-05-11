from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import timedelta, datetime
import os
import subprocess

def collect_data():
    """Run the script to collect weather data."""
    subprocess.run(['python', 'src/collect_data.py'], check=True)

def update_dvc():
    """Stage CSV file changes and push with DVC."""
    subprocess.run(['git', 'add', 'data/raw/glasgow_weather_data.csv'], check=True)
    subprocess.run(['git', 'commit', '-m', 'Append weather data from Airflow'], check=True)
    subprocess.run(['dvc', 'push'], check=True)

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2025, 5, 9),
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id='dvc_data_pipeline',
    default_args=default_args,
    description='A DVC-integrated pipeline that collects and pushes data every 10 minutes',
    schedule_interval="*/10 * * * *",  # Every 10 minutes
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
