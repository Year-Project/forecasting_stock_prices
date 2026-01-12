from uuid import UUID

from pydantic import BaseModel


class AdminInitResponse(BaseModel):
    user_id: UUID
    telegram_id: int
    secret: str

