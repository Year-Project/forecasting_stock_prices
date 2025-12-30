from pydantic import BaseModel


class ForecastPublishMessage(BaseModel):
    isin: str
    forecast_period: int
    time_frame: str
    forecast_price: float
    forecast_confidence: float | None = None
    forecast_plot: str | None = None

