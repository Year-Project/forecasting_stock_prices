from pydantic import BaseModel


class AdminSecretMessage(BaseModel):
    telegram_id: int
    secret: str
