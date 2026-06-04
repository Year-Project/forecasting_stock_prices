from typing import Annotated, Any
import logging
import os
import re

from fastapi import Depends
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX

from magician.dependencies import get_scavenger_client
from magician.schemas.request.get_forecast_request import GetForecastRequest
from magician.schemas.response.get_forecast_response import GetForecastResponse
from magician.schemas.response.get_candles_response import Candle
from magician.services.scavenger_client import ScavengerClient


MODEL_NAME = "auto_arima"
ISIN_PATTERN = re.compile(r"^[A-Z]{2}[A-Z0-9]{10}$")
logger = logging.getLogger(__name__)


class ForecastService:
    def __init__(self, scavenger_client: Annotated[ScavengerClient, Depends(get_scavenger_client)]):
        self._scavenger_client = scavenger_client
        self._ml_enabled = os.getenv("ML_FORECAST_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
        self._ml_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        self._week_model_uri = os.getenv("MLFLOW_WEEK_MODEL_URI", "models:/stock_return_forecaster_week@prd")
        self._month_model_uri = os.getenv("MLFLOW_MONTH_MODEL_URI", "models:/stock_return_forecaster_month@prd")
        self._ml_models: dict[str, Any] = {}

    async def get_forecast(self, request: GetForecastRequest) -> GetForecastResponse:
        interval = self._time_frame_to_interval(request.time_frame)

        candles = await self._get_candles(request, interval)

        model_uri = self._model_uri_for_request(interval, request.forecast_period)
        if model_uri is not None:
            try:
                forecast_return, model_label = self._ml_return_forecast(candles.candles, model_uri)
                forecast_price = self._last_observed_price(candles.candles) * float(np.exp(forecast_return))
                return GetForecastResponse(
                    isin=request.isin,
                    forecast_period=request.forecast_period,
                    time_frame=request.time_frame,
                    forecast_price=forecast_price,
                    forecast_return=forecast_return,
                    forecast_confidence=None,
                    forecast_plot=None,
                    model=model_label,
                )
            except Exception as exc:
                logger.exception("MLflow forecast failed; falling back to auto_arima", extra={"fallback_reason": str(exc)})

        forecast_price = self._auto_arima_forecast(candles.candles, request.forecast_period)

        return GetForecastResponse(
            isin=request.isin,
            forecast_period=request.forecast_period,
            time_frame=request.time_frame,
            forecast_price=forecast_price,
            forecast_return=None,
            forecast_confidence=None,
            forecast_plot=None,
            model=MODEL_NAME,
        )

    async def _get_candles(self, request: GetForecastRequest, interval: int):
        identifier = request.isin.strip().upper()
        if self._looks_like_isin(identifier):
            return await self._scavenger_client.get_candles(isin=identifier, interval=interval)
        return await self._scavenger_client.get_candles(ticker=identifier, interval=interval)

    def _model_uri_for_request(self, interval: int, forecast_period: int) -> str | None:
        if not self._ml_enabled or interval != 24:
            return None
        if forecast_period <= 0:
            raise ValueError("forecast_period must be positive.")
        if forecast_period <= 7:
            return self._week_model_uri
        if forecast_period <= 31:
            return self._month_model_uri
        return None

    def _ml_return_forecast(self, history: list[Candle], model_uri: str) -> tuple[float, str]:
        model = self._load_ml_model(model_uri)
        model_input = self._candles_to_model_input(history)
        prediction = model.predict(model_input)
        if not isinstance(prediction, pd.DataFrame):
            prediction = pd.DataFrame(prediction)
        if prediction.empty or "forecast_return" not in prediction.columns:
            raise ValueError("MLflow model did not return forecast_return")

        row = prediction.iloc[0]
        forecast_return = float(row["forecast_return"])
        ticker = str(row.get("ticker", model_input["ticker"].iloc[-1]))
        selected_model = str(row.get("model_name", "unknown"))
        model_label = f"{model_uri}:{selected_model}:{ticker}"
        logger.info(
            "MLflow forecast completed",
            extra={
                "ticker": ticker,
                "model_uri": model_uri,
                "selected_model": selected_model,
                "forecast_return": forecast_return,
            },
        )
        return forecast_return, model_label

    def _load_ml_model(self, model_uri: str):
        if model_uri not in self._ml_models:
            import mlflow

            if self._ml_tracking_uri:
                mlflow.set_tracking_uri(self._ml_tracking_uri)
            self._ml_models[model_uri] = mlflow.pyfunc.load_model(model_uri)
        return self._ml_models[model_uri]

    @staticmethod
    def _candles_to_model_input(history: list[Candle]) -> pd.DataFrame:
        rows = []
        for candle in sorted(history, key=lambda item: item.begin):
            if any(value is None for value in [candle.open, candle.high, candle.low, candle.close, candle.volume]):
                continue
            rows.append(
                {
                    "date": candle.begin,
                    "ticker": candle.ticker,
                    "open": candle.open,
                    "high": candle.high,
                    "low": candle.low,
                    "close": candle.close,
                    "volume": candle.volume,
                }
            )
        if not rows:
            raise ValueError("History has no complete OHLCV rows for ML forecast.")
        return pd.DataFrame(rows)

    @staticmethod
    def _last_observed_price(history: list[Candle]) -> float:
        for candle in sorted(history, key=lambda item: item.begin, reverse=True):
            if candle.close is not None:
                return float(candle.close)
            if candle.open is not None:
                return float(candle.open)
        raise ValueError("History has no price data.")

    @staticmethod
    def _looks_like_isin(identifier: str) -> bool:
        return bool(ISIN_PATTERN.fullmatch(identifier))

    @staticmethod
    def _time_frame_to_interval(time_frame: str) -> int:
        normalized = time_frame.strip().lower()
        mapping = {
            "1m": 1,
            "10m": 10,
            "1h": 60,
            "1d": 24,
            "1w": 7,
            "1mo": 31,
            "1q": 4,
        }
        if normalized in mapping:
            return mapping[normalized]
        if normalized.isdigit():
            return int(normalized)
        raise ValueError(f"Unsupported time_frame: {time_frame}")

    @staticmethod
    def _baseline_forecast(history: list[Candle], forecast_period: int) -> float:
        if not history:
            raise ValueError("No history returned from Scavenger.")

        closes = [candle.close for candle in history if candle.close is not None]
        if not closes:
            closes = [candle.open for candle in history if candle.open is not None]
        if not closes:
            raise ValueError("History has no price data.")

        window = max(1, min(len(closes), forecast_period))
        return float(sum(closes[-window:]) / window)

    @staticmethod
    def _auto_arima_forecast(history: list[Candle], forecast_period: int) -> float:
        if forecast_period <= 0:
            raise ValueError("forecast_period must be positive.")
        if not history:
            raise ValueError("No history returned from Scavenger.")

        closes = [candle.close for candle in history if candle.close is not None]
        if not closes:
            closes = [candle.open for candle in history if candle.open is not None]
        if not closes:
            raise ValueError("History has no price data.")

        series = np.asarray(closes, dtype=float)
        if series.size < 4:
            return ForecastService._baseline_forecast(history, forecast_period)

        d = 0
        max_d = 2
        while d <= max_d:
            diffed = np.diff(series, n=d) if d > 0 else series
            if diffed.size < 3:
                break
            try:
                p_value = adfuller(diffed, autolag="AIC")[1]
            except ValueError:
                break
            if p_value < 0.05:
                break
            d += 1

        best_aic = float("inf")
        best_order: tuple[int, int, int] | None = None
        max_p = 3
        max_q = 3
        for p in range(max_p + 1):
            for q in range(max_q + 1):
                if p == 0 and q == 0:
                    continue
                try:
                    model = SARIMAX(
                        series,
                        order=(p, d, q),
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    )
                    result = model.fit(disp=False)
                except Exception:
                    continue
                if result.aic < best_aic:
                    best_aic = result.aic
                    best_order = (p, d, q)

        if best_order is None:
            return ForecastService._baseline_forecast(history, forecast_period)

        model = SARIMAX(
            series,
            order=best_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        result = model.fit(disp=False)
        forecast = result.forecast(steps=forecast_period)
        last_value = forecast.iloc[-1] if hasattr(forecast, "iloc") else forecast[-1]
        return float(last_value)
