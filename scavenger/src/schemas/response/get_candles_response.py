from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel


class Candle(BaseModel):
    begin: datetime
    end: datetime
    ticker: str
    isin: str
    interval: int
    open: Optional[float] = None
    close: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    volume: Optional[float] = None


class GetCandlesResponse(BaseModel):
    candles: List[Candle]
