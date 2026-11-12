"""Build the altered test sets from the clean test set.

The clean test set is the base version. `apply_changes` always starts from a
copy of it, so each scenario is the base plus a named list of changes, and the
base can always be recovered by applying no changes. Every scenario goes
through the same variable in the pipeline, which keeps the runs comparable.
"""

import pandas as pd

from src import config


def apply_changes(base_test: pd.DataFrame, changes: list[str]) -> pd.DataFrame:
    """Return a copy of the base test set with each listed change applied.

    The changes are applied literally, as in the brief (43,823 becomes 3,823),
    so change A produces negative incomes for most counties. Values are not
    clipped, because out-of-range values are what monitoring should catch.
    """
    df = base_test.copy()
    for key in changes:
        column, delta = config.CHANGES[key]
        df[column] = df[column] + delta
    return df


def validation_table(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Mean, min, and max of each changed column in every scenario.

    This checks that each change was applied before looking at the
    Evidently reports.
    """
    columns = [column for column, _ in config.CHANGES.values()]
    rows = []
    for name, df in frames.items():
        row = {"scenario": name}
        for column in columns:
            row[f"{column}_mean"] = round(df[column].mean(), 2)
            row[f"{column}_min"] = round(df[column].min(), 2)
            row[f"{column}_max"] = round(df[column].max(), 2)
        rows.append(row)
    return pd.DataFrame(rows)
