"""Run the whole assignment end to end.

Steps:
1. Load and prepare the data, write the EDA summary, and split train/test.
2. Train the model, build the reference set (training split with out-of-fold
   predictions), and log training to MLflow.
3. For each scenario (clean, A, A+B, A+B+C): build it from the clean test set,
   score it, measure accuracy, run the Evidently report, and log to MLflow.
4. Write the comparison table and plots.

Usage, from the repository root:
    python pipeline.py
"""

import mlflow

from src import compare, config, data_prep, monitor, scenarios, train


def main() -> None:
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT)

    # ------------------------------------------------------------ 1. data
    raw = data_prep.load_raw()
    data_prep.eda_summary(raw).to_csv(config.RESULTS_DIR / "eda_summary.csv")
    df = data_prep.prepare(raw)
    features = data_prep.feature_columns(df)
    train_df, base_test = data_prep.split(df)
    print(f"rows: {len(df)}  train: {len(train_df)}  test: {len(base_test)}  features: {len(features)}")

    # ------------------------------------------------------------ 2. train
    with mlflow.start_run(run_name="train"):
        model, reference, cv_metrics = train.fit(train_df, features)
        mlflow.log_params(config.MODEL_PARAMS)
        mlflow.log_params({"n_features": len(features), "dropped": ",".join(config.DROP_COLUMNS)})
        mlflow.log_metrics({f"cv_{k}": v for k, v in cv_metrics.items()})
        mlflow.log_artifact(str(config.MODEL_DIR / "model.joblib"))
    print(f"cross-validated (out-of-fold) metrics on train: {cv_metrics}")

    # ------------------------------------------------------------ 3. scenarios
    ws, project = monitor.open_project()
    bounds = monitor.range_bounds(train_df)

    scored, rows = {}, []
    for name, changes in config.SCENARIOS.items():
        # Every scenario is rebuilt from the untouched clean test set.
        current = scenarios.apply_changes(base_test, changes)
        current = train.score(model, current, features)
        scored[name] = current

        metrics = train.regression_metrics(current[config.TARGET], current[config.PREDICTION])
        snapshot = monitor.run(name, changes, current, reference, features, bounds, ws, project)
        drift = monitor.summarize(snapshot)

        row = {
            "scenario": name,
            "changes": ", ".join(changes) or "none",
            **metrics,
            "prediction_mean": round(float(current[config.PREDICTION].mean()), 2),
            **drift,
        }
        rows.append(row)

        with mlflow.start_run(run_name=f"monitor_{monitor.file_slug(name)}"):
            mlflow.set_tags({"scenario": name, "changes": row["changes"]})
            mlflow.log_metrics({
                **metrics,
                "prediction_mean": row["prediction_mean"],
                "n_drifted_features": drift["n_drifted_features"],
                "prediction_drift": int(drift["prediction_drift"]),
                "n_failed_tests": drift["n_failed_tests"],
            })
            mlflow.log_artifact(str(config.REPORTS_DIR / f"{monitor.file_slug(name)}.html"))

        print(f"{name:>6}: RMSE {metrics['rmse']:.2f}  R² {metrics['r2']:.3f}  "
              f"drifted {drift['drifted_features']}  prediction drift {drift['prediction_drift']}  "
              f"failed tests {drift['n_failed_tests']}")

    # ------------------------------------------------------------ 4. compare
    compare.save_table(scenarios.validation_table(scored), "scenario_validation")
    table = compare.comparison_table(rows)
    compare.save_table(table, "comparison")
    compare.plot_predicted_vs_actual(scored)
    compare.plot_prediction_distribution(reference, scored)
    compare.plot_rmse(table)

    print(f"\nResults written to {config.RESULTS_DIR.name}/. "
          f"Evidently UI: evidently ui --workspace {config.WORKSPACE_DIR.name} --port 8000")


if __name__ == "__main__":
    main()
