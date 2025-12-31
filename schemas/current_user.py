from pydantic import BaseModel
from uuid import UUID

from schemas.user_role import UserRole


class CurrentUser(BaseModel):
    user_id: UUID
    telegram_id: int
    role: UserRole
