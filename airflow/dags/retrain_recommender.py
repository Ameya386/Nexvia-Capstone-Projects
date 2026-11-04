"""Step 7 - Airflow DAG: weekly retraining of the whole recommender pipeline.
Setup:  export PROJECT_DIR=/path/to/AIML-03 ; copy this file into $AIRFLOW_HOME/dags ; airflow standalone
"""
import os
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

PROJECT = os.environ.get("PROJECT_DIR", "/opt/AIML-03")
DATASET = os.environ.get("DATASET", "small")

with DAG("retrain_recommender", start_date=datetime(2026, 10, 1), schedule="@weekly", catchup=False,
         default_args={"retries": 1, "retry_delay": timedelta(minutes=5)}, tags=["recsys"]) as dag:
    run = lambda tid, cmd: BashOperator(task_id=tid, bash_command=f"cd {PROJECT} && python -m {cmd}")

    prep = run("prepare_data", f"src.data_prep --dataset {DATASET}")
    feats = run("spark_features", "src.spark_features")
    retrieval = run("train_retrieval", "src.train_retrieval")
    index = run("build_faiss_index", "src.build_index")
    ranker = run("train_rankers", "src.train_ranker")      # also refreshes model_repository/ for Triton
    evaluate = run("evaluate", "src.evaluate")
    ab = run("ab_test", "src.ab_test")

    prep >> feats >> retrieval >> index >> ranker >> evaluate >> ab
