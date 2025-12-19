from pydantic import BaseModel


class GetForecastRequest(BaseModel):
    isin: str
    forecast_period: int = 7
    time_frame: str = '1d'
    provide_plot: bool = False
