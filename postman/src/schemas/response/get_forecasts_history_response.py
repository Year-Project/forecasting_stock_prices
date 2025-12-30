from datetime import datetime
from uuid import UUID

from pydantic import BaseModel

from schemas.forecast_request_status import ForecastRequestStatus
from schemas.parsed_timeframe import ParsedTimeframe


class ForecastsHistory(BaseModel):
    id: UUID
    isin: str
    time_frame_interval: int
    time_frame_unit: str
    requested_plot: bool
    model: str | None
    user_id: UUID
    status: ForecastRequestStatus
    error: str | None
    used_cache: bool
    duration_ms: float | None
    created_at: datetime


class GetForecastsHistoryResponse(BaseModel):
    isin: str | None = None
    time_frame: ParsedTimeframe | None = None
    requested_plot: bool | None = None
    model: str | None = None
    user_id: UUID | None = None
    error: str | None = None
    used_cache: bool | None = None
    from_date: datetime | None = None
    to_date: datetime | None = None
    history: list[ForecastsHistory]
