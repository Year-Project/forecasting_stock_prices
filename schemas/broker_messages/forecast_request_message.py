from uuid import UUID

from pydantic import BaseModel

from schemas.parsed_timeframe import ParsedTimeframe


class ForecastRequestMessage(BaseModel):
    request_id: UUID
    isin: str
    forecast_period: int = 7
    time_frame: ParsedTimeframe
    provide_plot: bool = False
