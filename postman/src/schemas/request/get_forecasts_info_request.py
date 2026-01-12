from datetime import datetime
from uuid import UUID

from pydantic import BaseModel


class GetForecastsInfoRequest(BaseModel):
    isin: str | None = None
    time_frame: str | None = None
    requested_plot: bool | None = None
    model: str | None = None
    user_id: UUID | None = None
    telegram_id: int | None = None
    error: str | None = None
    used_cache: bool | None = None
    from_date: datetime | None = None
    to_date: datetime | None = None
