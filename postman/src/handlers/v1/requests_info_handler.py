from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession

from dependencies.dependencies import get_admin_user
from postman.src.dependencies import get_forecast_requests_session_maker, get_forecast_service
from postman.src.schemas.request.get_forecasts_info_request import GetForecastsInfoRequest
from postman.src.schemas.response.get_forecasts_history_response import GetForecastsHistoryResponse
from postman.src.schemas.response.get_forecasts_stats_response import GetForecastsStatsResponse
from postman.src.services.forecast_info_service import ForecastInfoService
from postman.src.services.forecast_service import ForecastService

router = APIRouter(prefix="/info/v1", tags=["forecast_requests_info"], dependencies=[Depends(get_admin_user)])


@router.post("/stats", response_model=GetForecastsStatsResponse)
async def get_stats_handler(request: GetForecastsInfoRequest,
                            forecast_info_service: Annotated[ForecastInfoService, Depends()],
                            forecast_requests_sb: async_sessionmaker[AsyncSession]
                            = Depends(get_forecast_requests_session_maker)):
    return await forecast_info_service.get_stats(forecast_requests_sb, request)


@router.post("/history", response_model=GetForecastsHistoryResponse)
async def get_history_handler(request: GetForecastsInfoRequest,
                              forecast_info_service: Annotated[ForecastInfoService, Depends()],
                              forecast_requests_sb: async_sessionmaker[AsyncSession]
                              = Depends(get_forecast_requests_session_maker)):
    return await forecast_info_service.get_history(forecast_requests_sb, request)


@router.delete("/history")
async def delete_history_handler(forecast_service: ForecastService = Depends(get_forecast_service),
                                 forecast_requests_sb: async_sessionmaker[AsyncSession]
                                 = Depends(get_forecast_requests_session_maker)):
    await forecast_service.clear_history(forecast_requests_sb)
