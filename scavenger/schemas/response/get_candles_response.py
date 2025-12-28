from datetime import datetime
from typing import Optional, List
from uuid import UUID
from pydantic import BaseModel


class Candle(BaseModel):
    begin: datetime
    end: datetime
    ticker: str
    interval: int
    open: Optional[float] = None
    close: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    volume: Optional[float] = None


class GetCandlesResponse(BaseModel):
    ticker: Optional[str] = None
    interval: Optional[int] = None
    from_date: Optional[datetime] = None
    to_date: Optional[datetime] = None
    history: List[Candle]
