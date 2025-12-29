from pydantic import BaseModel
from uuid import UUID

from schemas.user_role import UserRole


class CurrentUser(BaseModel):
    user_id: UUID
    role: UserRole
