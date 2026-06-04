import joblib
import numpy as np
import pandas as pd
import pytest

from stock_forecast.artifacts import save_json
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
