from datetime import date
from pydantic import BaseModel
from typing import Optional

class GetCandlesRequest(BaseModel):
    ticker: str
    interval: int
    start: Optional[date] = None
    end: Optional[date] = None
