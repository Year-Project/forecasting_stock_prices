from __future__ import annotations

from pathlib import Path
from typing import Any
import re

import joblib
import numpy as np
import pandas as pd

from .artifacts import load_json
from .features import make_features

try:  # pragma: no cover - exercised when MLflow is installed in runtime image
    from mlflow.pyfunc import PythonModel
except ImportError:  # pragma: no cover - keeps local unit tests importable without MLflow
    PythonModel = object  # type: ignore[misc,assignment]


MODEL_BUNDLE_ARTIFACT = "bundle"
REQUIRED_INPUT_COLUMNS = ("date", "ticker", "open", "high", "low", "close", "volume")


class StockReturnPyFuncRouter(PythonModel):
    """Route each ticker to the strict-protocol final model selected for it."""

    def load_context(self, context: Any) -> None:
        bundle_path = Path(context.artifacts[MODEL_BUNDLE_ARTIFACT])
        self.load_bundle(bundle_path)

    def load_bundle(self, bundle_path: str | Path) -> None:
        bundle = Path(bundle_path)
        metadata = load_json(bundle / "metadata.json")
        selected_models = load_json(bundle / "selected_models.json")
        feature_payload = load_json(bundle / "feature_columns.json")

        self.horizon_name = str(metadata["horizon_name"])
        self.horizon = int(metadata["horizon"])
        self.target_column = str(feature_payload.get("target_column", ""))
        self.default_feature_cols = list(feature_payload["feature_columns"])
        self.models_by_ticker: dict[str, dict[str, Any]] = {}

        for row in selected_models:
            ticker = str(row["ticker"])
            model_name = str(row["model_name"])
            model_path = bundle / "models" / safe_filename(model_name) / safe_filename(ticker) / "final.pkl"
            payload = joblib.load(model_path)
            estimator = payload["estimator"] if isinstance(payload, dict) and "estimator" in payload else payload
            feature_cols = list(payload.get("feature_cols", self.default_feature_cols)) if isinstance(payload, dict) else self.default_feature_cols
            self.models_by_ticker[ticker] = {
                "model_name": model_name,
                "estimator": estimator,
                "feature_cols": feature_cols,
                "metadata": payload if isinstance(payload, dict) else {},
            }

    def predict(self, context, model_input, params=None) -> pd.DataFrame:
        frame = _to_dataframe(model_input)
        _validate_input_frame(frame)
        frame = frame.copy()
        frame["date"] = pd.to_datetime(frame["date"])
        frame["ticker"] = frame["ticker"].astype(str)

        feature_frame = make_features(frame).sort_values(["ticker", "date"])
        rows: list[dict[str, Any]] = []
        for ticker in _input_tickers_in_order(frame):
            if ticker not in self.models_by_ticker:
                raise ValueError(f"No registered forecast model for ticker {ticker}")

            model_info = self.models_by_ticker[ticker]
            ticker_features = feature_frame[feature_frame["ticker"].astype(str) == ticker].sort_values("date").copy()
            if ticker_features.empty:
                raise ValueError(f"No feature rows available for ticker {ticker}")

            feature_cols = list(model_info["feature_cols"])
            for col in feature_cols:
                if col not in ticker_features.columns:
                    ticker_features[col] = np.nan

            estimator = model_info["estimator"]
            if _requires_full_frame(model_info):
                prediction_input = ticker_features[["date", "ticker", *feature_cols]].copy()
            else:
                prediction_input = ticker_features.tail(1).loc[:, feature_cols].copy()

            prediction = np.asarray(estimator.predict(prediction_input), dtype=float)
            forecast_return = float(prediction[-1])
            latest_raw = frame[frame["ticker"].astype(str) == ticker].sort_values("date").tail(1)
            rows.append(
                {
                    "ticker": ticker,
                    "horizon_name": self.horizon_name,
                    "horizon": self.horizon,
                    "model_name": model_info["model_name"],
                    "forecast_return": forecast_return,
                    "prediction_date": latest_raw["date"].iloc[0],
                }
            )
        return pd.DataFrame(rows)


def safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._") or "value"


def _requires_full_frame(model_info: dict[str, Any]) -> bool:
    if str(model_info.get("model_name", "")).lower() == "lstm":
        return True
    estimator = model_info.get("estimator")
    return bool(hasattr(estimator, "training_context_"))


def _to_dataframe(model_input: Any) -> pd.DataFrame:
    if isinstance(model_input, pd.DataFrame):
        return model_input
    if isinstance(model_input, dict):
        if all(not isinstance(value, (list, tuple, pd.Series, np.ndarray)) for value in model_input.values()):
            return pd.DataFrame([model_input])
        return pd.DataFrame(model_input)
    return pd.DataFrame(model_input)


def _validate_input_frame(frame: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_INPUT_COLUMNS if col not in frame.columns]
    if missing:
        raise ValueError(f"Missing model input columns: {missing}")
    if frame.empty:
        raise ValueError("Model input is empty")


def _input_tickers_in_order(frame: pd.DataFrame) -> list[str]:
    ordered = frame.sort_values(["ticker", "date"]).drop_duplicates("ticker", keep="last")
    return [str(ticker) for ticker in ordered["ticker"]]
