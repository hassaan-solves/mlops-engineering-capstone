"""Drift-triggered retraining for fraud-detector.

One command closes the loop: detect drift -> (only if drifted) retrain
on the combined data -> register the new run as a `fraud-detector`
version -> move the `production` alias to it. This is the "automated
retraining" the lab is about -- no manual clicking in the MLflow UI.

The plumbing to run the provided drift/retrain scripts and to find the
new run id is written for you. Author the two TODOs.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import mlflow

HERE = Path(__file__).resolve().parent
TRACKING_URI = "http://localhost:5000"
MODEL_NAME = "fraud-detector"
ALIAS = "production"

mlflow.set_tracking_uri(TRACKING_URI)
client = mlflow.tracking.MlflowClient()


def _run(script: str) -> None:
    """Run one of the provided scripts (drift.py / retrain.py)."""
    subprocess.check_call([sys.executable, str(HERE / script)])


def _latest_retrain_run_id() -> str:
    exp = client.get_experiment_by_name("fraud-detection")
    runs = client.search_runs(
        [exp.experiment_id],
        filter_string="tags.mlflow.runName = 'retrain'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise SystemExit("no `retrain` run found after retraining")
    return runs[0].info.run_id


# --- 1. Detect drift (writes reports/drift-summary.json + drift.html).
_run("drift.py")
summary = json.loads((HERE / "reports" / "drift-summary.json").read_text())
drifted = bool(summary.get("dataset_drift"))
print(f"[loop] dataset_drift={drifted}")


if drifted:
    _run("retrain.py")
else:
    print ("no drift found")
    sys.exit(0)

# --- 2. Retrain on the combined data (logs a `retrain` run to MLflow).
run_id = _latest_retrain_run_id()
print(f"[loop] retrained: run_id={run_id}")

model_uri = f"runs:/{run_id}/model"
regsitered_model = mlflow.register_model(
    model_uri,
    MODEL_NAME)

client.set_registered_model_alias(
    MODEL_NAME,
    ALIAS,
    regsitered_model.version)
    
