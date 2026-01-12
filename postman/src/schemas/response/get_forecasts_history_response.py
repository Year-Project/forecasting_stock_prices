from datetime import datetime
from uuid import UUID

from pydantic import BaseModel

from schemas.forecast_request_status import ForecastRequestStatus


class ForecastsHistory(BaseModel):
    id: UUID
    isin: str
    time_frame: str
    requested_plot: bool
    model: str | None = None
    user_id: UUID
    telegram_id: int | None = None
    status: ForecastRequestStatus
    error: str | None = None
    used_cache: bool
    duration_ms: float | None = None
    created_at: datetime


class GetForecastsHistoryResponse(BaseModel):
    isin: str | None = None
    time_frame: str | None = None
    requested_plot: bool | None = None
    model: str | None = None
    user_id: UUID | None = None
    error: str | None = None
    used_cache: bool | None = None
    from_date: datetime | None = None
    to_date: datetime | None = None
    history: list[ForecastsHistory] | None = None
