from datetime import datetime, timedelta
import asyncio

import numpy as np
import pandas as pd
import pytest

from magician.schemas.request.get_forecast_request import GetForecastRequest
from magician.schemas.response.get_candles_response import Candle, GetCandlesResponse
from magician.services.forecast_service import ForecastService


class FakeMlflowModel:
    def __init__(self, forecast_return: float):
        self.forecast_return = forecast_return
        self.input_frame = None

    def predict(self, model_input):
        self.input_frame = model_input
        return pd.DataFrame(
            [
                {
                    "ticker": model_input["ticker"].iloc[-1],
                    "model_name": "ridge",
                    "forecast_return": self.forecast_return,
                }
            ]
        )


class FakeScavengerClient:
    def __init__(self):
        self.calls = []

    async def get_candles(self, **kwargs):
        self.calls.append(kwargs)
        start = datetime(2026, 1, 1)
        candles = []
        for idx in range(5):
            begin = start + timedelta(days=idx)
            close = 100.0 + idx
            candles.append(
                Candle(
                    begin=begin,
                    end=begin + timedelta(days=1),
                    ticker="SBER",
                    isin="",
                    interval=24,
                    open=close - 0.5,
                    high=close + 1.0,
                    low=close - 1.0,
                    close=close,
                    volume=1000 + idx,
                )
            )
        return GetCandlesResponse(candles=candles)


class FakeForecastService(ForecastService):
    def __init__(self, scavenger_client, fake_model):
        super().__init__(scavenger_client)
        self.fake_model = fake_model
        self.loaded_uris = []

    def _load_ml_model(self, model_uri: str):
        self.loaded_uris.append(model_uri)
        return self.fake_model


def test_daily_week_request_uses_mlflow_return_and_price(monkeypatch):
    monkeypatch.setenv("ML_FORECAST_ENABLED", "true")
    monkeypatch.setenv("MLFLOW_WEEK_MODEL_URI", "models:/week@prd")
    fake_model = FakeMlflowModel(forecast_return=0.1)
    scavenger = FakeScavengerClient()
    service = FakeForecastService(scavenger, fake_model)

    response = asyncio.run(
        service.get_forecast(
            GetForecastRequest(isin="SBER", forecast_period=7, time_frame="1d", provide_plot=False)
        )
    )

    assert response.forecast_return == pytest.approx(0.1)
    assert response.forecast_price == pytest.approx(104.0 * np.exp(0.1))
    assert response.model == "models:/week@prd:ridge:SBER"
    assert service.loaded_uris == ["models:/week@prd"]
    assert scavenger.calls[0]["ticker"] == "SBER"
    assert fake_model.input_frame["ticker"].iloc[-1] == "SBER"


def test_non_daily_request_falls_back_to_auto_arima(monkeypatch):
    monkeypatch.setenv("ML_FORECAST_ENABLED", "true")
    monkeypatch.setattr(ForecastService, "_auto_arima_forecast", staticmethod(lambda history, forecast_period: 123.0))
    fake_model = FakeMlflowModel(forecast_return=0.1)
    service = FakeForecastService(FakeScavengerClient(), fake_model)

    response = asyncio.run(
        service.get_forecast(
            GetForecastRequest(isin="SBER", forecast_period=7, time_frame="1w", provide_plot=False)
        )
    )

    assert response.forecast_price == 123.0
    assert response.forecast_return is None
    assert response.model == "auto_arima"
    assert service.loaded_uris == []
