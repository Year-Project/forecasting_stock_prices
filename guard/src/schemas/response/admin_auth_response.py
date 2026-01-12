from pydantic import BaseModel


class AdminAuthResponse(BaseModel):
    access_token: str

