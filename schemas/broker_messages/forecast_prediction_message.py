from uuid import UUID

from pydantic import BaseModel

from schemas.parsed_timeframe import ParsedTimeframe


class ForecastPredictionMessage(BaseModel):
    isin: str
    forecast_period: int
    time_frame: ParsedTimeframe
    forecast_price: float
    forecast_confidence: float | None = None
    forecast_plot: str | None = None

