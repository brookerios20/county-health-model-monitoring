"""Load the cancer dataset, drop unusable columns, split, and summarize.

The test split is created once with a fixed seed and is used only for
evaluation and monitoring, never for training or model choices.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

from src import config


def load_raw(path=config.DATA_PATH) -> pd.DataFrame:
    """Read the CSV exactly as provided.

    The file is Latin-1 encoded: some county names in `Geography` contain
    characters such as ñ (byte 0xF1), which are invalid as UTF-8. The target
    column is renamed as set in `config.SOURCE_RENAMES`.
    """
    return pd.read_csv(path, encoding="latin-1").rename(columns=config.SOURCE_RENAMES)


def prepare(raw: pd.DataFrame) -> pd.DataFrame:
    """Drop the identifier, the derived income bin, and the mostly empty column."""
    return raw.drop(columns=config.DROP_COLUMNS)


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Every remaining column except the target is a model input."""
    return [c for c in df.columns if c not in (config.TARGET, config.PREDICTION)]


def split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train, test) with a fixed seed so reruns give the same rows."""
    train, test = train_test_split(df, test_size=config.TEST_SIZE, random_state=config.SEED)
    return train.reset_index(drop=True), test.reset_index(drop=True)


def eda_summary(raw: pd.DataFrame) -> pd.DataFrame:
    """One row per numeric column: missing values, range, and correlation with the target."""
    numeric = raw.select_dtypes("number")
    summary = pd.DataFrame({
        "missing": numeric.isna().sum(),
        "missing_share": numeric.isna().mean().round(3),
        "min": numeric.min(),
        "median": numeric.median(),
        "max": numeric.max(),
        "corr_with_target": numeric.corrwith(numeric[config.TARGET]).round(3),
    })
    summary.index.name = "column"
    return summary
