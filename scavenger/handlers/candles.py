import asyncio
from fastapi import APIRouter, Depends, HTTPException

from services.moex.moex import CandlesService, get_candles_service
from schemas.response.get_candles_response import GetCandlesResponse, Candle
from schemas.request.get_candles_request import GetCandlesRequest

router = APIRouter(prefix="/info/v1", tags=["candles"])

@router.get("/", response_model=GetCandlesResponse, summary="Get MOEX candles")
async def get_candles_handler(
    request: GetCandlesRequest = Depends(),
    candles_service: CandlesService = Depends(get_candles_service)
):
    try:
        loop = asyncio.get_running_loop()
        df = await loop.run_in_executor(
            None,
            candles_service.fetch_candles,
            request.ticker,
            request.interval,
            request.start,
            request.end
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if df.empty:
        return GetCandlesResponse(
            ticker=request.ticker,
            interval=request.interval,
            from_date=request.start,
            to_date=request.end,
            history=[]
        )

    history = [
        Candle(
            begin=row["begin"],
            end=row["end"],
            ticker=row["ticker"],
            interval=row["interval"],
            open=row.get("open"),
            close=row.get("close"),
            high=row.get("high"),
            low=row.get("low"),
            volume=row.get("volume"),
        )
        for _, row in df.iterrows()
    ]

    return GetCandlesResponse(
        ticker=request.ticker,
        interval=request.interval,
        from_date=request.start,
        to_date=request.end,
        history=history
    )