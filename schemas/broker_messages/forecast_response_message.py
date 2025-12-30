from uuid import UUID

from pydantic import BaseModel

from schemas.forecast_request_status import ForecastRequestStatus


class ForecastResponseMessage(BaseModel):
    request_id: UUID
    isin: str
    forecast_period: int
    time_frame: str
    forecast_price: float
    forecast_confidence: float | None = None
    forecast_plot: str | None = None
    provide_plot: bool = False
    model: str | None = None
    status: ForecastRequestStatus
    error: str | None = None

