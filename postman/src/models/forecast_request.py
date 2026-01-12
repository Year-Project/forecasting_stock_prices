from datetime import datetime, timezone
from uuid import UUID, uuid4

from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import UUID as UUID_ORM, DateTime, String, Boolean, Float, Enum as Enum_ORM, Integer

from postman.src.models.base import Base
from schemas.forecast_request_status import ForecastRequestStatus


class ForecastRequest(Base):
    __tablename__ = "forecast_requests"

    id: Mapped[UUID] = mapped_column(UUID_ORM(as_uuid=True), primary_key=True, default=uuid4)
    isin: Mapped[str] = mapped_column(String, nullable=False, index=True)
    time_frame: Mapped[str] = mapped_column(String, nullable=False)
    requested_plot: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    model: Mapped[str | None] = mapped_column(String, nullable=False, default='Undefined')
    user_id: Mapped[UUID | None] = mapped_column(UUID_ORM(as_uuid=True), nullable=True, index=True)
    telegram_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    status: Mapped[ForecastRequestStatus] = mapped_column(Enum_ORM(ForecastRequestStatus, name='status_enum'),
                                                          nullable=False)
    error: Mapped[str | None] = mapped_column(String, nullable=True)
    used_cache: Mapped[bool] = mapped_column(Boolean, default=False)
    duration_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False,
                                                 default=lambda: datetime.now(timezone.utc), index=True)
