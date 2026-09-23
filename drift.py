"""Detect drift between `/root/code/data/reference.csv` and
`/root/code/data/current.csv` using Evidently, then save:

- `/root/code/reports/drift.html` — the interactive HTML report
  (browseable via the Drift Report button).
- `/root/code/reports/drift-summary.json` — a small machine-readable
  summary with `dataset_drift` (bool) and `number_of_drifted_columns`.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.metrics import DriftedColumnsCount, ValueDrift

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
REPORTS = HERE / "reports"
REPORTS.mkdir(exist_ok=True, parents=True)

reference = pd.read_csv(DATA / "reference.csv")
current = pd.read_csv(DATA / "current.csv")

# Per-column drift + the dataset-level drifted-columns share/count.
columns = list(reference.columns)
metrics = [ValueDrift(column=c) for c in columns] + [DriftedColumnsCount()]

ref_ds = Dataset.from_pandas(reference, data_definition=DataDefinition())
cur_ds = Dataset.from_pandas(current, data_definition=DataDefinition())

result = Report(metrics).run(current_data=cur_ds, reference_data=ref_ds)
result.save_html(str(REPORTS / "drift.html"))

# Pull the drifted-columns share + count out of the DriftedColumnsCount
# metric in the report's serialised result.
share = 0.0
drifted_columns = 0
for m in result.dict().get("metrics", []):
    name = str(m.get("metric_name") or m.get("metric_id") or "")
    value = m.get("value")
    if "DriftedColumnsCount" in name and isinstance(value, dict):
        share = float(value.get("share", 0.0))
        drifted_columns = int(float(value.get("count", 0)))

# Dataset drift = a majority of columns drifted (Evidently's default share
# threshold). The seeded reference vs current shift every feature, so this
# is True for this lab's data.
summary = {
    "dataset_drift": bool(share >= 0.5),
    "number_of_drifted_columns": drifted_columns,
}
(REPORTS / "drift-summary.json").write_text(json.dumps(summary, indent=2))

print(f"Drift report saved to {REPORTS / 'drift.html'}")
print(
    f"Dataset drift: {summary['dataset_drift']}. "
    f"Drifted columns: {summary['number_of_drifted_columns']}."
)
if summary["dataset_drift"]:
    print(
        "Drift detected. Run retrain.py to train a new model on the "
        "union of reference + current."
    )