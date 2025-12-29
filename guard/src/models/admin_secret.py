from datetime import datetime, timezone
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import UUID as UUID_ORM, DateTime, String, ForeignKey

from guard.src.models.base import Base


if TYPE_CHECKING:
    from .user import User


class AdminSecret(Base):
    __tablename__ = "admin_secret"

    id: Mapped[UUID] = mapped_column(UUID_ORM(as_uuid=True), primary_key=True, default=uuid4)
    user_id: Mapped[UUID] = mapped_column(UUID_ORM(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    user: Mapped["User"] = relationship(back_populates="admin_secret")
    secret_hash: Mapped[str] = mapped_column(String, nullable=False, unique=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False,
                                                 default=lambda: datetime.now(timezone.utc))
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

