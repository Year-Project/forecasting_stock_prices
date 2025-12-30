from uuid import UUID

from pydantic import BaseModel


class ForecastResponseMessage(BaseModel):
    request_id: UUID
    isin: str
    forecast_period: int
    time_frame: str
    forecast_price: float | None = None
    forecast_confidence: float | None = None
    forecast_plot: str | None = None
    provide_plot: bool = False
    model: str | None = None
    status: str
    error: str | None = None
