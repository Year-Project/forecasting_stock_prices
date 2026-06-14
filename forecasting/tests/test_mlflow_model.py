from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from stock_forecast.artifacts import save_json, save_table
from stock_forecast.mlflow_cli import _model_training_run_payloads
from stock_forecast.mlflow_tracking import MLFLOW_PARENT_RUN_ID, MLflowRunConfig, log_strict_protocol_result, mlflow_horizon_run
from stock_forecast.mlflow_model import StockReturnPyFuncRouter


class ConstantEstimator:
    def __init__(self, value: float):
        self.value = value

    def predict(self, X):
        return np.full(len(X), self.value, dtype=float)


def _write_bundle(tmp_path, ticker="SBER", value=0.05):
    bundle = tmp_path / "bundle"
    model_dir = bundle / "models" / "ridge" / ticker
    model_dir.mkdir(parents=True)
    joblib.dump(
        {
            "estimator": ConstantEstimator(value),
            "model_name": "ridge",
            "ticker": ticker,
            "feature_cols": ["ret_1"],
        },
        model_dir / "final.pkl",
    )
    save_json(
        [{"ticker": ticker, "model_name": "ridge", "validation_directional_accuracy": 0.6}],
        bundle / "selected_models.json",
    )
    save_json(
        {
            "horizon_name": "week",
            "horizon": 5,
            "feature_columns": ["ret_1"],
            "target_column": "target_return_5_next_open",
        },
        bundle / "feature_columns.json",
    )
    save_json({"horizon_name": "week", "horizon": 5}, bundle / "metadata.json")
    return bundle


def _ohlcv_frame(ticker="SBER"):
    dates = pd.date_range("2026-01-01", periods=5, freq="D")
    return pd.DataFrame(
        {
            "date": dates,
            "ticker": ticker,
            "open": [100, 101, 102, 103, 104],
            "high": [101, 102, 103, 104, 105],
            "low": [99, 100, 101, 102, 103],
            "close": [100.5, 101.5, 102.5, 103.5, 104.5],
            "volume": [1000, 1100, 1200, 1300, 1400],
        }
    )


def test_pyfunc_router_predicts_selected_ticker_return(tmp_path):
    router = StockReturnPyFuncRouter()
    router.load_bundle(_write_bundle(tmp_path, value=0.07))

    prediction = router.predict(None, _ohlcv_frame())

    assert prediction.loc[0, "ticker"] == "SBER"
    assert prediction.loc[0, "model_name"] == "ridge"
    assert prediction.loc[0, "horizon_name"] == "week"
    assert prediction.loc[0, "forecast_return"] == pytest.approx(0.07)


def test_pyfunc_router_rejects_unsupported_ticker(tmp_path):
    router = StockReturnPyFuncRouter()
    router.load_bundle(_write_bundle(tmp_path, ticker="SBER"))

    with pytest.raises(ValueError, match="No registered forecast model"):
        router.predict(None, _ohlcv_frame(ticker="VTBR"))


def test_mlflow_training_run_payloads_include_results_and_best_params(tmp_path):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    save_table(
        pd.DataFrame(
            [
                {
                    "ticker": "SBER",
                    "model_name": "ridge",
                    "n_obs": 20,
                    "directional_accuracy": 0.65,
                    "rmse": 0.02,
                    "n_train": 120,
                    "split_quality": "mature",
                    "limited_history": False,
                }
            ]
        ),
        reports_dir / "validation_prediction_metrics.parquet",
    )
    save_table(
        pd.DataFrame(
            [
                {
                    "ticker": "SBER",
                    "model_name": "ridge",
                    "n_obs": 20,
                    "directional_accuracy": 0.7,
                    "rmse": 0.015,
                    "validation_rank": 1,
                    "validation_directional_accuracy": 0.65,
                }
            ]
        ),
        reports_dir / "test_prediction_metrics.parquet",
    )
    save_table(
        pd.DataFrame(
            [
                {
                    "ticker": "SBER",
                    "model_name": "ridge",
                    "validation_rank": 1,
                    "validation_directional_accuracy": 0.65,
                    "is_validation_selected": True,
                }
            ]
        ),
        reports_dir / "validation_model_ranking.parquet",
    )
    save_table(
        pd.DataFrame([{"ticker": "SBER", "model_name": "ridge"}]),
        reports_dir / "selected_models_by_ticker.parquet",
    )
    save_table(
        pd.DataFrame(
            [
                {
                    "ticker": "SBER",
                    "model_name": "ridge",
                    "signal_mode": "overlapping_tranches",
                    "sharpe": 1.4,
                    "cumulative_return": 0.12,
                }
            ]
        ),
        reports_dir / "test_signal_metrics.parquet",
    )
    save_json(
        [{"ticker": "SBER", "model_name": "ridge", "best_params": {"alpha": 1.0}}],
        reports_dir / "best_params.json",
    )

    payloads = _model_training_run_payloads(reports_dir, "week", 5)

    assert len(payloads) == 1
    payload = payloads[0]
    assert payload["tags"]["is_validation_selected"] == "true"
    assert payload["params"]["best_param_alpha"] == 1.0
    assert payload["metrics"]["validation_directional_accuracy"] == pytest.approx(0.65)
    assert payload["metrics"]["test_directional_accuracy"] == pytest.approx(0.7)
    assert payload["metrics"]["selection_validation_rank"] == pytest.approx(1.0)
    assert payload["metrics"]["test_signal_overlapping_tranches_sharpe"] == pytest.approx(1.4)


def test_mlflow_trial_logger_creates_nested_trial_run(tmp_path):
    mlflow = pytest.importorskip("mlflow")
    from mlflow.tracking import MlflowClient

    tracking_uri = (tmp_path / "mlruns").as_uri()
    experiment_name = "trial-logging-test"
    config = MLflowRunConfig(
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        notebook_name="test_notebook",
        horizon_name="week",
        horizon=5,
        enabled=True,
        log_optuna_trials=True,
    )

    with mlflow_horizon_run(config, params={"primary_metric": "directional_accuracy"}) as handle:
        assert handle.run_id is not None
        assert handle.trial_logger is not None
        parent_run_id = handle.run_id
        handle.trial_logger.log_trial(
            trial_number=3,
            model_name="ridge",
            params={"alpha": 1.0, "solver": "svd"},
            metrics={"objective_score": 0.62, "directional_accuracy": 0.62},
            tags={"ticker": "SBER", "objective_metric": "directional_accuracy"},
        )

    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    runs = client.search_runs([experiment.experiment_id])
    child_runs = [run for run in runs if run.info.run_id != parent_run_id]

    assert len(child_runs) == 1
    child = child_runs[0]
    assert child.data.tags[MLFLOW_PARENT_RUN_ID] == parent_run_id
    assert child.data.tags["ticker"] == "SBER"
    assert child.data.params["alpha"] == "1.0"
    assert child.data.params["solver"] == "svd"
    assert child.data.metrics["objective_score"] == pytest.approx(0.62)
    assert child.data.metrics["directional_accuracy"] == pytest.approx(0.62)
    assert mlflow.get_tracking_uri() == tracking_uri


def test_mlflow_result_logger_writes_metrics_tables_and_artifacts(tmp_path):
    pytest.importorskip("mlflow")
    from mlflow.tracking import MlflowClient

    reports_dir = tmp_path / "reports"
    models_dir = tmp_path / "models" / "ridge" / "SBER"
    reports_dir.mkdir()
    models_dir.mkdir(parents=True)
    (reports_dir / "summary.txt").write_text("report", encoding="utf-8")
    (models_dir / "final.pkl").write_bytes(b"model")

    validation_metrics = pd.DataFrame(
        [{"ticker": "SBER", "model_name": "ridge", "rmse": 0.02, "directional_accuracy": 0.65}]
    )
    test_signal_metrics = pd.DataFrame(
        [{"ticker": "SBER", "model_name": "ridge", "signal_mode": "overlapping_tranches", "sharpe": 1.4}]
    )
    result = {
        "reports_dir": reports_dir,
        "models_dir": tmp_path / "models",
        "validation_prediction_metrics": validation_metrics,
        "test_signal_metrics": test_signal_metrics,
        "leakage_audit": pd.DataFrame([{"check": "ok", "passed": True}]),
    }

    tracking_uri = (tmp_path / "mlruns").as_uri()
    experiment_name = "result-logging-test"
    config = MLflowRunConfig(
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        notebook_name="test_notebook",
        horizon_name="week",
        horizon=5,
        enabled=True,
    )
    with mlflow_horizon_run(config) as handle:
        parent_run_id = handle.run_id
        log_strict_protocol_result(result, params={"target_col": "target_return_5_next_open"})

    client = MlflowClient()
    parent = client.get_run(parent_run_id)
    table_artifacts = {artifact.path for artifact in client.list_artifacts(parent_run_id, "tables")}
    report_artifacts = {artifact.path for artifact in client.list_artifacts(parent_run_id, "reports")}
    model_artifacts = {artifact.path for artifact in client.list_artifacts(parent_run_id, "models/ridge/SBER")}

    assert parent.data.params["target_col"] == "target_return_5_next_open"
    assert parent.data.metrics["validation_rmse_mean"] == pytest.approx(0.02)
    assert parent.data.metrics["test_signal_sharpe_mean"] == pytest.approx(1.4)
    assert "tables/validation_prediction_metrics.json" in table_artifacts
    assert "reports/summary.txt" in report_artifacts
    assert "models/ridge/SBER/final.pkl" in model_artifacts


def test_training_notebooks_enable_mlflow_tracking():
    notebook_paths = [
        "forecasting/notebooks/02_table_model_forecasting.ipynb",
        "forecasting/notebooks/02b_lstm_forecasting.ipynb",
        "forecasting/notebooks/02c_global_lstm_forecasting.ipynb",
    ]
    for path in notebook_paths:
        notebook = Path(path).read_text()
        assert "MLFLOW_ENABLED" in notebook
        assert "MLFLOW_TRACKING_URI" in notebook
        assert "MLFLOW_LOG_OPTUNA_TRIALS" in notebook
        assert "MLflowRunConfig" in notebook
        assert "mlflow_horizon_run" in notebook
        assert "mlflow_trial_logger=mlflow_run.trial_logger" in notebook
        assert "log_strict_protocol_result" in notebook
        assert "register_model" not in notebook
