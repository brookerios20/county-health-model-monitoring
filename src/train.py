"""Train the cancer mortality-rate model and build the monitoring reference set.

The reference set for Evidently is the training split with out-of-fold
predictions. Out-of-fold predictions come from models that never saw the row,
so their error is comparable with the error on the test set. In-sample
predictions would make the reference look more accurate than any new data.
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict

from src import config


def build_model() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(**config.MODEL_PARAMS)


def regression_metrics(y_true, y_pred) -> dict:
    """RMSE, MAE, and R² rounded for reporting."""
    return {
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 3),
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 3),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
    }


def fit(train: pd.DataFrame, features: list[str]):
    """Fit the final model and return (model, reference set, CV metrics).

    The reference set is the training split plus a `prediction` column of
    out-of-fold predictions from 5-fold cross-validation.
    """
    X, y = train[features], train[config.TARGET]
    folds = KFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.SEED)
    oof = cross_val_predict(build_model(), X, y, cv=folds)

    model = build_model().fit(X, y)

    reference = train.copy()
    reference[config.PREDICTION] = oof

    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, config.MODEL_DIR / "model.joblib")

    return model, reference, regression_metrics(y, oof)


def score(model, df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Return a copy of df with the model's predictions in the `prediction` column."""
    scored = df.copy()
    scored[config.PREDICTION] = model.predict(df[features])
    return scored
