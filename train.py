"""Train a fraud-detector model and log the run + artefacts to MLflow.

Reads the dataset from the SeaweedFS `data` bucket, trains a RandomForest
on [amount, hour, num_tx_past_day] to predict `is_fraud`, and logs
everything to the `fraud-detection` experiment on the local MLflow
tracking server (backend: SQLite, artefacts: SeaweedFS).
"""
from __future__ import annotations

import os
from io import BytesIO
from pathlib import Path

import boto3
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
cfg = yaml.safe_load((HERE / "config.yaml").read_text())

os.environ["AWS_ACCESS_KEY_ID"] = cfg["s3"]["access_key"]
os.environ["AWS_SECRET_ACCESS_KEY"] = cfg["s3"]["secret_key"]
os.environ["MLFLOW_S3_ENDPOINT_URL"] = cfg["s3"]["endpoint"]

s3 = boto3.client(
    "s3",
    endpoint_url=cfg["s3"]["endpoint"],
    aws_access_key_id=cfg["s3"]["access_key"],
    aws_secret_access_key=cfg["s3"]["secret_key"],
)
obj = s3.get_object(
    Bucket=cfg["s3"]["data_bucket"], Key="transactions.csv",
)
df = pd.read_csv(BytesIO(obj["Body"].read()))

X = df.drop(columns=["is_fraud"])
y = df["is_fraud"]
X_train, _, y_train, _ = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42,
)

mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
mlflow.set_experiment("fraud-detection")
mlflow.sklearn.autolog()

with mlflow.start_run() as run:
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print(f"Logged MLflow run id={run.info.run_id}")