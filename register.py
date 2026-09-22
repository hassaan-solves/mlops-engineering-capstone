"""Register the trained run as `fraud-detector` and promote it.

train.py logs a run to the `fraud-detection` experiment. This script
turns that run into the model the serving layer targets: it registers
`runs:/<run_id>/model` as a version of `fraud-detector`, then puts the
`production` alias on that version. serve.py resolves
`models:/fraud-detector@production`, so this alias is the handoff from
training to serving.
"""
from __future__ import annotations

import mlflow

TRACKING_URI = "http://localhost:5000"
MODEL_NAME = "fraud-detector"
ALIAS = "production"

mlflow.set_tracking_uri(TRACKING_URI)
client = mlflow.tracking.MlflowClient()


def _latest_run_id() -> str:
    """The most recent run id in the `fraud-detection` experiment."""
    exp = client.get_experiment_by_name("fraud-detection")
    if exp is None:
        raise SystemExit("no `fraud-detection` experiment yet -- run train.py first")
    runs = client.search_runs(
        [exp.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise SystemExit("no runs in `fraud-detection` -- run train.py first")
    return runs[0].info.run_id


run_id = _latest_run_id()
print(f"[register] latest run_id={run_id}")


model_uri = f"runs:/{run_id}/model"

result = mlflow.register_model(
    model_uri=model_uri,
    name=MODEL_NAME,
)

client.set_registered_model_alias(
    name=MODEL_NAME,
    alias=ALIAS,
    version=result.version,
)

print(f"promoted {MODEL_NAME} version {result.version} successfully")