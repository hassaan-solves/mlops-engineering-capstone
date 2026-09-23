"""Retrain the fraud-detector on the union of reference + current.

Logs a new MLflow run to the `fraud-detection` experiment (autolog).
Registration + promotion are handled automatically by
retrain_if_drift.py — this script only trains and logs the run.
"""
from __future__ import annotations

import os
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

os.environ["AWS_ACCESS_KEY_ID"] = "weedadmin"
os.environ["AWS_SECRET_ACCESS_KEY"] = "weedadmin123"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:8333"

ref = pd.read_csv(DATA / "reference.csv")
cur = pd.read_csv(DATA / "current.csv")
df = pd.concat([ref, cur], ignore_index=True)

X = df.drop(columns=["is_fraud"])
y = df["is_fraud"]
X_train, _, y_train, _ = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42,
)

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("fraud-detection")
mlflow.sklearn.autolog()

with mlflow.start_run(run_name="retrain") as run:
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    print(f"Logged retrain run id={run.info.run_id}")