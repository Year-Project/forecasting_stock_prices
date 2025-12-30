from typing import Annotated

from fastapi import Depends
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX

from magician.dependencies import get_scavenger_client
from magician.schemas.request.get_forecast_request import GetForecastRequest
from magician.schemas.response.get_forecast_response import GetForecastResponse
from magician.schemas.response.get_candles_response import Candle
from magician.services.scavenger_client import ScavengerClient


class ForecastService:
    def __init__(self, scavenger_client: Annotated[ScavengerClient, Depends(get_scavenger_client)]):
        self._scavenger_client = scavenger_client

    async def get_forecast(self, request: GetForecastRequest) -> GetForecastResponse:
        interval = self._time_frame_to_interval(request.time_frame)

        candles = await self._scavenger_client.get_candles(
            ticker=request.isin,
            interval=interval,
        )

        forecast_price = self._auto_arima_forecast(candles.history, request.forecast_period)

        return GetForecastResponse(
            isin=request.isin,
            forecast_period=request.forecast_period,
            time_frame=request.time_frame,
            forecast_price=forecast_price,
            forecast_confidence=None,
            forecast_plot=None,
        )

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
