import asyncio
from fastapi import APIRouter, Depends, HTTPException
from typing import List

from scavenger.src.services.moex.moex import MOEXService, get_moex_service
from scavenger.src.schemas.response.get_candles_response import Candle
from scavenger.src.schemas.request.get_candles_request import GetCandlesRequest

router = APIRouter(prefix="/info/v1", tags=["candles"])

@router.get("/candles", response_model=List[Candle], summary="Get MOEX candles by ticker or ISIN")
async def get_candles_handler(
    request: GetCandlesRequest = Depends(),
    candles_service: MOEXService = Depends(get_moex_service)
):
    try:
        loop = asyncio.get_running_loop()
        
        df = await loop.run_in_executor(
            None,
            lambda: candles_service.fetch_candles(
                isin=request.isin,
                ticker=request.ticker,
                interval=request.interval,
                start=request.start,
                end=request.end
            )
        )

        return [
            Candle(
                begin=row["begin"],
                end=row["end"],
                ticker=row["ticker"],
                isin=row.get("original_isin", request.isin or ""),
                interval=row["interval"],
                open=row.get("open"),
                close=row.get("close"),
                high=row.get("high"),
                low=row.get("low"),
                volume=row.get("volume"),
            )
            for _, row in df.iterrows()
        ]
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
