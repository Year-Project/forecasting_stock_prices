from uuid import UUID

from pydantic import BaseModel


class ForecastRequestMessage(BaseModel):
    request_id: UUID
    isin: str
    forecast_period: int = 7
    time_frame: str
    provide_plot: bool = False
