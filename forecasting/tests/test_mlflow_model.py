import joblib
import numpy as np
import pandas as pd
import pytest

from stock_forecast.artifacts import save_json, save_table
from stock_forecast.mlflow_cli import _model_training_run_payloads
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
