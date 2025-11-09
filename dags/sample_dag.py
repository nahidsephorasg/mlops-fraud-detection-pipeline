from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from datetime import datetime, timedelta


# Simple Python functions
def print_hello():
    print("Hello from Airflow!")
    return "Hello task completed"


def print_date():
    print(f"Current date and time: {datetime.now()}")
    return "Date task completed"


def print_goodbye():
    print("Goodbye from Airflow!")
    return "Goodbye task completed"


# Default arguments
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2025, 11, 8),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}

# Define the DAG
with DAG(
    "sample_hello_world_dag",
    default_args=default_args,
    description="A simple hello world DAG for testing",
    schedule=None,  # Manual trigger only
    catchup=False,
    tags=["sample", "test", "hello_world"],
) as dag:

    # Task 1: Print hello
    hello_task = PythonOperator(task_id="print_hello", python_callable=print_hello)

    # Task 2: Print date
    date_task = PythonOperator(task_id="print_date", python_callable=print_date)

    # Task 3: Print goodbye
    goodbye_task = PythonOperator(
        task_id="print_goodbye", python_callable=print_goodbye
    )

    # Define task dependencies
    hello_task >> date_task >> goodbye_task
