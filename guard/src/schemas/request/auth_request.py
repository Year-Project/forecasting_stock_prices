from pydantic import BaseModel


class AuthRequest(BaseModel):
    telegram_id: int
