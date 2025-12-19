from typing import Annotated

from fastapi import APIRouter, Depends
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession

from Postman.main import get_session_maker, get_forecast_request_producer
from Postman.src.schemas.Request.GetForecastRequest import GetForecastRequest
from Postman.src.services.forecast_service import ForecastService
from db.cache import get_redis_client
from kafka.BaseBrokerProducer import BaseBrokerProducer

router = APIRouter(prefix="/forecasts", tags=["forecasts"])


@router.post("/get_forecast")
async def get_forecast_handler(request: GetForecastRequest, forecast_service: Annotated[ForecastService, Depends()],
                               forecast_requests_sb: async_sessionmaker[AsyncSession]
                               = Depends(get_session_maker("forecast_requests")),
                               redis_client: Redis = Depends(get_redis_client),
                               broker_producer: BaseBrokerProducer = Depends(get_forecast_request_producer)):
    await forecast_service.get_forecasts(forecast_requests_sb, redis_client, request)
