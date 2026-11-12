"""Evidently monitoring: one report definition, run once per scenario.

Every run compares a scored current dataset (the clean test set or one of the
altered copies) with the same reference dataset (the training split with
out-of-fold predictions). Each report contains, in this order:

1. Data drift on the input features (DataDriftPreset).
2. Drift on the model output (ValueDrift on the `prediction` column).
3. Regression quality (RegressionPreset), comparing predictions with the target.
4. Range tests on the three changed columns, which turn the result into
   pass or fail alerts. Bounds come from the training data, and medIncome
   also has a rule that income cannot be negative.

Uses the Evidently 0.7 API, as in the week 7 lab.
"""

import json
import re

import pandas as pd
from evidently import DataDefinition, Dataset, Regression, Report
from evidently.metrics import MaxValue, MinValue, ValueDrift
from evidently.presets import DataDriftPreset, RegressionPreset
from evidently.sdk.models import PanelMetric
from evidently.sdk.panels import line_plot_panel
from evidently.tests import gte, lte
from evidently.ui.workspace import Workspace

from src import config


def range_bounds(train: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """Training min and max of each changed column, used as test bounds."""
    columns = [column for column, _ in config.CHANGES.values()]
    return {c: (float(train[c].min()), float(train[c].max())) for c in columns}


def to_dataset(df: pd.DataFrame, features: list[str]) -> Dataset:
    """Wrap a scored DataFrame with the schema Evidently needs."""
    definition = DataDefinition(
        numerical_columns=features + [config.TARGET, config.PREDICTION],
        regression=[Regression(target=config.TARGET, prediction=config.PREDICTION)],
    )
    return Dataset.from_pandas(df, data_definition=definition)


def build_report(features: list[str], bounds: dict[str, tuple[float, float]]) -> Report:
    """The single report definition shared by every run."""
    range_tests = []
    for column, (low, high) in bounds.items():
        min_tests = [gte(low)]
        if column == "medIncome":
            min_tests.insert(0, gte(0))  # business rule: income cannot be negative
        range_tests.append(MinValue(column=column, tests=min_tests))
        range_tests.append(MaxValue(column=column, tests=[lte(high)]))

    return Report([
        DataDriftPreset(
            columns=features,
            drift_share=config.DATASET_DRIFT_SHARE,
            num_method=config.DRIFT_METHOD,
            threshold=config.DRIFT_THRESHOLD,
        ),
        ValueDrift(
            column=config.PREDICTION,
            method=config.DRIFT_METHOD,
            threshold=config.DRIFT_THRESHOLD,
        ),
        RegressionPreset(),
        *range_tests,
    ])


def open_project():
    """Open the local workspace and the project, creating either if needed."""
    config.WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        ws = Workspace.create(str(config.WORKSPACE_DIR))
    except Exception:
        ws = Workspace(str(config.WORKSPACE_DIR))

    # Reuse the project on reruns so all runs stay in one project.
    try:
        for project in ws.list_projects():
            if project.name == config.EVIDENTLY_PROJECT:
                return ws, project
    except Exception:
        pass

    project = ws.create_project(config.EVIDENTLY_PROJECT)
    project.description = (
        "County cancer mortality-rate model. Reference: training split with out-of-fold "
        "predictions. Runs: clean test set, then changes A, A+B, and A+B+C."
    )
    project.save()
    add_dashboard(project)  # only for a new project, so reruns add no duplicate panels
    return ws, project


def add_dashboard(project) -> None:
    """Three line panels that track the monitoring signals across runs.

    Each panel reads one metric from every run in the project, so the four
    scenarios appear as four points in run order.
    """
    project.dashboard.add_panel(line_plot_panel(
        title="Drifted input features",
        description="Number of the 30 input features with K-S p < 0.05 against the reference",
        values=[PanelMetric(legend="drifted features", metric="DriftedColumnsCount",
                            metric_labels={"value_type": "count"})],
    ))
    project.dashboard.add_panel(line_plot_panel(
        title="Prediction drift (K-S p-value)",
        description="Below 0.05 means the model output has drifted from the reference",
        values=[PanelMetric(legend="prediction p-value", metric="ValueDrift",
                            metric_labels={"column": config.PREDICTION})],
    ))
    project.dashboard.add_panel(line_plot_panel(
        title="Test RMSE",
        description="Per 100,000 population; the reference out-of-fold RMSE is about 15.9",
        values=[PanelMetric(legend="RMSE", metric="RMSE")],
    ))


def run(name: str, changes: list[str], current: pd.DataFrame, reference: pd.DataFrame,
        features: list[str], bounds: dict, ws, project):
    """Run the report for one scenario, save HTML and JSON, and log it to the workspace.

    The run is named and tagged with its scenario, so the Reports list in the
    Evidently UI shows which test set each run used.
    """
    report = build_report(features, bounds)
    described = [f"{key}: {config.CHANGES[key][0]} {config.CHANGES[key][1]:+g}" for key in changes]
    snapshot = report.run(
        to_dataset(current, features),
        to_dataset(reference, features),
        name=name,
        tags=[name],
        metadata={"scenario": name, "changes": "; ".join(described) or "none"},
    )

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    slug = file_slug(name)
    snapshot.save_html(str(config.REPORTS_DIR / f"{slug}.html"))
    snapshot.save_json(str(config.REPORTS_DIR / f"{slug}.json"))
    ws.add_run(project.id, snapshot, include_data=False, name=name)
    return snapshot


def file_slug(name: str) -> str:
    """'A+B+C' -> 'a_plus_b_plus_c', used for report file names."""
    return name.lower().replace("+", "_plus_")


# ---------------------------------------------------------------- summarizing
# The snapshot's dict layout is not documented in detail, so the helpers below
# read it defensively and the raw JSON is saved next to each HTML report.

def _metric_name(entry: dict) -> str:
    for key in ("metric_id", "metric_name", "name", "id"):
        if isinstance(entry.get(key), str):
            return entry[key]
    return ""


def _column_of(name: str) -> str | None:
    match = re.search(r"column=([^,)]+)", name)
    return match.group(1) if match else None


def summarize(snapshot) -> dict:
    """Pull drift results and failed tests out of one snapshot.

    Returns the drifted input features, the prediction drift p-value and flag,
    and the descriptions of failed tests.
    """
    result = json.loads(snapshot.json())
    drifted, prediction_p = [], None

    for entry in result.get("metrics", []):
        name = _metric_name(entry)
        if not name.startswith("ValueDrift"):
            continue
        column, value = _column_of(name), entry.get("value")
        if column is None or not isinstance(value, (int, float)):
            continue
        if column == config.PREDICTION:
            prediction_p = float(value)
        elif value < config.DRIFT_THRESHOLD:
            drifted.append(column)

    failed = [
        t.get("description") or t.get("name") or str(t)
        for t in result.get("tests", [])
        if str(t.get("status", "")).upper() in ("FAIL", "FAILED", "ERROR")
    ]

    return {
        "drifted_features": drifted,
        "n_drifted_features": len(drifted),
        "prediction_drift_p": prediction_p,
        "prediction_drift": prediction_p is not None and prediction_p < config.DRIFT_THRESHOLD,
        "failed_tests": failed,
        "n_failed_tests": len(failed),
    }
