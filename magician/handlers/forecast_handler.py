from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from magician.schemas.request.get_forecast_request import GetForecastRequest
from magician.schemas.response.get_forecast_response import GetForecastResponse
from magician.services.forecast_service import ForecastService

router = APIRouter(prefix="/forecasts/v1", tags=["forecasts"])


@router.post("/forecast", response_model=GetForecastResponse)
async def get_forecast_handler(
    request: GetForecastRequest,
    forecast_service: Annotated[ForecastService, Depends()],
):
    try:
        return await forecast_service.get_forecast(request)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
