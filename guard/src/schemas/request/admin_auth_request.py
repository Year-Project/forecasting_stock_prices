from pydantic import BaseModel


class AdminAuthRequest(BaseModel):
    telegram_id: int
    secret_key: str
