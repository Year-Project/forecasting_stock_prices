from datetime import datetime, timezone
from enum import Enum
from uuid import UUID, uuid4

from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import UUID as UUID_ORM, DateTime, String, Boolean, Float, Enum as Enum_ORM

from Postman.src.models.Base import Base


class ForecastRequestStatus(str, Enum):
    COMPLETED = 'completed'
    PENDING = 'pending'
    FAILED = 'failed'


class ForecastRequest(Base):
    __tablename__ = "forecast_requests"

    id: Mapped[UUID] = mapped_column(UUID_ORM(as_uuid=True), primary_key=True, default=uuid4)
    isin: Mapped[str] = mapped_column(String, nullable=False)
    time_frame: Mapped[str] = mapped_column(String, nullable=False)
    requested_plot: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    model: Mapped[str | None] = mapped_column(String, nullable=False, default='Undefined')
    user_id: Mapped[UUID] = mapped_column(UUID_ORM(as_uuid=True))
    status: Mapped[ForecastRequestStatus] = mapped_column(Enum_ORM(ForecastRequestStatus, name='status_enum'),
                                                          nullable=False)
    error: Mapped[str | None] = mapped_column(String, nullable=False, default=None)
    used_cache: Mapped[bool] = mapped_column(Boolean, default=False)
    duration_ms: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False,
                                                 default=lambda: datetime.now(timezone.utc))
