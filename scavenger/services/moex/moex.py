from typing import Optional
from datetime import date
import pandas as pd
import requests
import apimoex


class CandlesService:
    @staticmethod
    def interval_to_timedelta(interval: int) -> pd.Timedelta:
        match interval:
            case 1:
                return pd.Timedelta(minutes=1)
            case 10:
                return pd.Timedelta(minutes=10)
            case 60:
                return pd.Timedelta(hours=1)
            case 24:
                return pd.Timedelta(days=1)
            case 7:
                return pd.Timedelta(weeks=1)
            case 31:
                return pd.Timedelta(days=31)
            case 4:
                return pd.Timedelta(days=91)
            case _:
                raise ValueError(f"Unsupported interval: {interval}")

    def fetch_candles(
        self,
        ticker: str,
        interval: int,
        start: Optional[date],
        end: Optional[date],
    ) -> pd.DataFrame:
        with requests.Session() as session:
            candles = apimoex.get_market_candles(
                session=session,
                security=ticker,
                interval=interval,
                start=start,
                end=end,
                market="shares",
            )
        df = pd.DataFrame(candles)
        if df.empty:
            return df

        df["begin"] = pd.to_datetime(df["begin"])
        df["ticker"] = ticker
        df["interval"] = interval
        df["end"] = df["begin"] + self.interval_to_timedelta(interval)
        return df


def get_candles_service() -> CandlesService:
    return CandlesService()