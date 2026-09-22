"""FastAPI inference server for the fraud-detector model.

Boots immediately. A background loader polls the MLflow Model
Registry for ``models:/<model_name>@<model_alias>`` and swaps the
loaded model in once it becomes available, so the server is up
before the registered model exists or the alias is assigned.
"""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import mlflow
import mlflow.pyfunc
import pandas as pd
import uvicorn
import yaml
from fastapi import FastAPI, HTTPException

FEATURE_COLUMNS = ["amount", "hour", "num_tx_past_day"]

HERE = Path(__file__).resolve().parent
cfg = yaml.safe_load((HERE / "config.yaml").read_text())

os.environ["AWS_ACCESS_KEY_ID"] = cfg["s3"]["access_key"]
os.environ["AWS_SECRET_ACCESS_KEY"] = cfg["s3"]["secret_key"]
os.environ["MLFLOW_S3_ENDPOINT_URL"] = cfg["s3"]["endpoint"]

mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])

model_uri = (
    f"models:/{cfg['mlflow']['model_name']}@{cfg['mlflow']['model_alias']}"
)

app = FastAPI(title="fraud-detector serve")
_state = {"model": None}


def _loader():
    while _state["model"] is None:
        try:
            _state["model"] = mlflow.pyfunc.load_model(model_uri)
            print(f"[serve] loaded {model_uri}")
            return
        except Exception:
            time.sleep(5)


threading.Thread(target=_loader, daemon=True).start()


@app.get("/health")
def health():
    status = "healthy" if _state["model"] is not None else "loading"
    return {"status": status, "model_uri": model_uri}


@app.post("/predict")
def predict(payload: dict):
    model = _state["model"]
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=(
                f"model {model_uri} is not yet available. Register the "
                "fraud-detection run as `fraud-detector` in the MLflow "
                "UI and assign the `production` alias."
            ),
        )
    values = payload["features"]
    df = pd.DataFrame([dict(zip(FEATURE_COLUMNS, values))])
    df["amount"] = df["amount"].astype("float64")
    df["hour"] = df["hour"].astype("int64")
    df["num_tx_past_day"] = df["num_tx_past_day"].astype("int64")
    pred = int(model.predict(df)[0])
    return {"prediction": pred}


if __name__ == "__main__":
    uvicorn.run(app, host=cfg["api"]["host"], port=cfg["api"]["port"])