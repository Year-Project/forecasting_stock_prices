from datetime import datetime, timezone
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import UUID as UUID_ORM, DateTime, BigInteger, Enum as Enum_ORM

from schemas.user_role import UserRole

from guard.src.models.base import Base

if TYPE_CHECKING:
    from .admin_secret import AdminSecret
    from .refresh_token import RefreshToken


class User(Base):
    __tablename__ = "users"

    id: Mapped[UUID] = mapped_column(UUID_ORM(as_uuid=True), primary_key=True, default=uuid4)
    telegram_id: Mapped[int] = mapped_column(BigInteger, nullable=False, unique=True, index=True)
    role: Mapped[UserRole] = mapped_column(Enum_ORM(UserRole, name='user_role_enum'), nullable=False,
                                           default=UserRole.USER)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False,
                                                 default=lambda: datetime.now(timezone.utc))
    refresh_tokens: Mapped[list["RefreshToken"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    admin_secret: Mapped["AdminSecret"] = relationship(back_populates="user")

