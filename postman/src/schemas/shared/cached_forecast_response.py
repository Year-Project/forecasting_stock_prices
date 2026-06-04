from pydantic import BaseModel


class CachedForecastResponse(BaseModel):
    isin: str
    forecast_period: int
    time_frame: str
    forecast_price: float
    forecast_return: float | None = None
    model: str
    provide_plot: bool = False
    forecast_confidence: float | None = None
    forecast_plot: str | None = None
