from datetime import date
from typing import Optional

import httpx

from magician.schemas.response.get_candles_response import GetCandlesResponse


class ScavengerClient:
    def __init__(self, client: httpx.AsyncClient):
        self._client = client

    async def get_candles(
        self,
        ticker: str,
        interval: int,
        start: Optional[date] = None,
        end: Optional[date] = None,
    ) -> GetCandlesResponse:
        params: dict[str, str | int] = {
            "ticker": ticker,
            "interval": interval,
        }
        if start is not None:
            params["start"] = start.isoformat()
        if end is not None:
            params["end"] = end.isoformat()

        response = await self._client.get("/scavenger/info/v1/", params=params)
        response.raise_for_status()
        return GetCandlesResponse.model_validate(response.json())
