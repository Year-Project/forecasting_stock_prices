from typing import Annotated

from fastapi import APIRouter, Depends, Response, status
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession

from db.cache import get_redis_client
from kafka.BaseBrokerProducer import BaseBrokerProducer
from postman.src.dependencies import get_forecast_requests_session_maker, get_forecast_request_producer
from postman.src.schemas.request.get_forecast_request import GetForecastRequest
from postman.src.schemas.response.get_forecast_response import GetForecastResponse
from postman.src.services.forecast_service import ForecastService

router = APIRouter(prefix="/forecasts/v1", tags=["forecasts"])


@router.post("/forecast", response_model=GetForecastResponse | None)
async def get_forecast_handler(request: GetForecastRequest, response: Response,
                               forecast_service: Annotated[ForecastService, Depends()],
                               forecast_requests_sb: async_sessionmaker[AsyncSession]
                               = Depends(get_forecast_requests_session_maker),
                               redis_client: Redis = Depends(get_redis_client),
                               broker_producer: BaseBrokerProducer = Depends(get_forecast_request_producer)):

    result = await forecast_service.get_forecasts(forecast_requests_sb, redis_client, request, broker_producer)

    if result is None:
        response.status_code = status.HTTP_201_CREATED
    else:
        response.status_code = status.HTTP_200_OK

    return result
