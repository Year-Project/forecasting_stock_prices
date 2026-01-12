from datetime import date
from pydantic import BaseModel, field_validator, model_validator
from typing import Optional

class GetCandlesRequest(BaseModel):
    ticker: Optional[str] = None
    isin: Optional[str] = None
    interval: int
    start: Optional[date] = None
    end: Optional[date] = None

    @field_validator('ticker', 'isin')
    @classmethod
    def strip_whitespace(cls, v: Optional[str]) -> Optional[str]:
        if v:
            return v.strip()
        return v

    @model_validator(mode='after')
    def validate_at_least_one_identifier(self) -> 'GetCandlesRequest':
        if not self.ticker and not self.isin:
            raise ValueError('Either "ticker" or "isin" must be provided')
        if self.ticker and self.isin:
            raise ValueError('Provide only one: either "ticker" or "isin"')
        return self