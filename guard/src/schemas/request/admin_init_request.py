from pydantic import BaseModel


class AdminInitRequest(BaseModel):
    telegram_id: int
