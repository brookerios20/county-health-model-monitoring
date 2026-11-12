"""Combine the four runs into one table and draw the comparison plots."""

import matplotlib

matplotlib.use("Agg")  # write files only, no display needed
import matplotlib.pyplot as plt
import pandas as pd

from src import config


def comparison_table(rows: list[dict]) -> pd.DataFrame:
    """One row per scenario with accuracy, prediction shift, drift, and alerts."""
    table = pd.DataFrame(rows)
    table["drifted_features"] = table["drifted_features"].apply(", ".join)
    table["failed_tests"] = table["failed_tests"].apply(" | ".join)
    return table


def save_table(table: pd.DataFrame, stem: str) -> None:
    """Write a table as both CSV and Markdown."""
    table.to_csv(config.RESULTS_DIR / f"{stem}.csv", index=False)
    (config.RESULTS_DIR / f"{stem}.md").write_text(table.to_markdown(index=False) + "\n")


def plot_predicted_vs_actual(scored: dict[str, pd.DataFrame]) -> None:
    """Predicted against actual mortality rate, one panel per scenario, shared axes."""
    fig, axes = plt.subplots(1, len(scored), figsize=(4 * len(scored), 4), sharex=True, sharey=True)
    for ax, (name, df) in zip(axes, scored.items()):
        ax.scatter(df[config.TARGET], df[config.PREDICTION], s=8, alpha=0.5)
        low, high = df[config.TARGET].min(), df[config.TARGET].max()
        ax.plot([low, high], [low, high], color="grey", linewidth=1)
        ax.set_title(name)
        ax.set_xlabel("actual mortality rate")
    axes[0].set_ylabel("predicted mortality rate")
    fig.suptitle("Predicted vs actual mortality rate on the test set (per 100,000 population)")
    fig.tight_layout()
    fig.savefig(config.RESULTS_DIR / "predicted_vs_actual.png", dpi=150)
    plt.close(fig)


def plot_prediction_distribution(reference: pd.DataFrame, scored: dict[str, pd.DataFrame]) -> None:
    """Distribution of model output for the reference set and each scenario."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(reference[config.PREDICTION], bins=40, density=True, alpha=0.35,
            label="reference (train, out-of-fold)")
    for name, df in scored.items():
        ax.hist(df[config.PREDICTION], bins=40, density=True, histtype="step", linewidth=1.5, label=name)
    ax.set_xlabel("predicted mortality rate")
    ax.set_ylabel("density")
    ax.set_title("Model output by scenario")
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.RESULTS_DIR / "prediction_distribution.png", dpi=150)
    plt.close(fig)


def plot_rmse(table: pd.DataFrame) -> None:
    """Test RMSE per scenario."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(table["scenario"], table["rmse"])
    for i, value in enumerate(table["rmse"]):
        ax.text(i, value, f"{value:.1f}", ha="center", va="bottom")
    ax.set_ylabel("RMSE (per 100,000 population)")
    ax.set_title("Test RMSE by scenario")
    fig.tight_layout()
    fig.savefig(config.RESULTS_DIR / "rmse_by_scenario.png", dpi=150)
    plt.close(fig)
