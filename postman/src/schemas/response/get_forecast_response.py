from pydantic import BaseModel


class GetForecastResponse(BaseModel):
    isin: str
    forecast_period: int
    time_frame: str
    telegram_id: int | None = None
    forecast_price: float | None = None
    forecast_confidence: float | None = None
    forecast_plot: str | None = None
