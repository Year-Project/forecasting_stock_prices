from typing import Optional
from datetime import date
import pandas as pd
import requests
import apimoex
import logging

logger = logging.getLogger(__name__)

class MOEXService:
    def get_ticker_by_isin(self, isin: str) -> str:
        try:
            with requests.Session() as session:
                data = apimoex.find_securities(
                    session=session,
                    string=isin,
                    columns=("secid", "isin", "regnumber", "name")
                )
                
                if not data:
                    logger.warning(f"No securities found for ISIN: {isin}")
                    raise ValueError(f"No securities found for ISIN: {isin}")
                
                first_secid = data[0].get("secid", "")
                logger.warning(f"Using {first_secid} for {isin}")
                return first_secid
                
        except Exception as e:
            logger.error(f"Error getting ticker for ISIN {isin}: {str(e)}")
            raise

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
        isin: Optional[str] = None,
        ticker: Optional[str] = None,
        interval: int = 24,
        start: Optional[date] = None,
        end: Optional[date] = None,
    ) -> pd.DataFrame:
        if isin:
            final_ticker = self.get_ticker_by_isin(isin)
            if not final_ticker:
                raise ValueError(f"Ticker not found for ISIN: {isin}")
        else:
            final_ticker = ticker
        
        logger.info(f"Fetching candles for {final_ticker}, interval {interval}")
        
        try:
            with requests.Session() as session:
                candles = apimoex.get_market_candles(
                    session=session,
                    security=final_ticker,
                    interval=interval,
                    start=start,
                    end=end,
                    market="shares",
                )
        except Exception as e:
            logger.error(f"Error fetching candles for {final_ticker}: {str(e)}")
            raise
        
        df = pd.DataFrame(candles)
        if df.empty:
            logger.warning(f"No candles found for {final_ticker}")
            return df

        df["begin"] = pd.to_datetime(df["begin"])
        df["ticker"] = final_ticker
        df["isin"] = isin
        df["interval"] = interval
        df["end"] = df["begin"] + self.interval_to_timedelta(interval)
        
        return df


def get_moex_service() -> MOEXService:
    return MOEXService()
