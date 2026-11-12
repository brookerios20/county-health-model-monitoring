"""Paths and settings shared by every stage of the pipeline.

Anything a reader might want to change lives here. Paths can be overridden with
environment variables so the project runs from any location without edits.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = Path(os.getenv("CANCER_CSV", ROOT / "data" / "cancer_reg.csv"))
RESULTS_DIR = Path(os.getenv("RESULTS_DIR", ROOT / "results"))
REPORTS_DIR = RESULTS_DIR / "evidently_reports"   # one HTML and one JSON per run
MODEL_DIR = RESULTS_DIR / "models"                # fitted model (excluded from git)
WORKSPACE_DIR = Path(os.getenv("EVIDENTLY_WORKSPACE", ROOT / "evidently_workspace"))

# ---------------------------------------------------------------- tracking
EVIDENTLY_PROJECT = "County cancer outcomes monitoring"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MLFLOW_EXPERIMENT = "cancer-outcomes-monitoring"

# ---------------------------------------------------------------- data
# The source CSV calls the target TARGET_deathRate. It is renamed on load to
# mortality_rate, the standard public-health term, and every later stage uses
# the new name.
SOURCE_RENAMES = {"TARGET_deathRate": "mortality_rate"}
TARGET = "mortality_rate"
PREDICTION = "prediction"

# Columns removed before modeling. The reasons are documented in the README:
#   Geography        county name, a unique identifier per row
#   binnedInc        decile bin of medIncome; change A would leave it out of sync
#   PctSomeCol18_24  missing for most rows
DROP_COLUMNS = ["Geography", "binnedInc", "PctSomeCol18_24"]

SEED = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# ---------------------------------------------------------------- model
# HistGradientBoostingRegressor handles missing values natively, so the
# pipeline needs no imputation step.
MODEL_PARAMS = {
    "max_iter": 300,
    "learning_rate": 0.05,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 20,
    "random_state": SEED,
}

# ---------------------------------------------------------------- test-set changes
# Each change adds a fixed amount to one column of the clean test set.
CHANGES = {
    "A": ("medIncome", -40_000),       # median income decreased by 40,000
    "B": ("povertyPercent", 20),       # poverty percent increased by 20 points
    "C": ("AvgHouseholdSize", 2),      # average household size increased by 2
}

# Runs in the order they are monitored. Every run is built from the clean test
# set, so the base version is never modified.
SCENARIOS = {
    "clean": [],
    "A": ["A"],
    "A+B": ["A", "B"],
    "A+B+C": ["A", "B", "C"],
}

# ---------------------------------------------------------------- monitoring
# The drift method is fixed so every run uses the same test. With the
# Kolmogorov-Smirnov test, a column is flagged as drifted when p < threshold.
DRIFT_METHOD = "ks"
DRIFT_THRESHOLD = 0.05

# Share of drifted input features at which Evidently reports the whole dataset
# as drifted. Evidently's default is 0.5. One drifted feature out of 30 (0.033)
# is enough here, because a single broken upstream column is the failure this
# project monitors for.
DATASET_DRIFT_SHARE = 0.03
