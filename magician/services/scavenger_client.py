from datetime import date
from typing import Optional

import httpx

from magician.schemas.response.get_candles_response import GetCandlesResponse


class ScavengerClient:
    def __init__(self, client: httpx.AsyncClient):
        self._client = client

    async def get_candles(
        self,
        ticker: str | None = None,
        interval: int = 24,
        start: Optional[date] = None,
        end: Optional[date] = None,
        isin: str | None = None,
    ) -> GetCandlesResponse:
        if bool(ticker) == bool(isin):
            raise ValueError("Provide exactly one identifier: ticker or isin.")

        params: dict[str, str | int] = {"interval": interval}
        if ticker is not None:
            params["ticker"] = ticker
        if isin is not None:
            params["isin"] = isin
        if start is not None:
            params["start"] = start.isoformat()
        if end is not None:
            params["end"] = end.isoformat()

        response = await self._client.get("/scavenger/info/v1/candles", params=params)
        response.raise_for_status()
        return GetCandlesResponse.model_validate(response.json())
