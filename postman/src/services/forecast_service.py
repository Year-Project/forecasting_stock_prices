import time
import hashlib

from typing import Annotated

from fastapi import Depends
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession

from kafka.BaseBrokerProducer import BaseBrokerProducer
from postman.src.models.forecast_request import ForecastRequest, ForecastRequestStatus
from postman.src.repositories.forecast_requests_repository import ForecastRequestsRepository
from postman.src.schemas.request.get_forecast_request import GetForecastRequest
from postman.src.schemas.response.get_forecast_response import GetForecastResponse
from postman.src.schemas.shared.cached_forecast_response import CachedForecastResponse


class ForecastService:
    def __init__(self, forecast_repository: Annotated[ForecastRequestsRepository, Depends()]):
        self._forecast_repository = forecast_repository

    async def get_forecasts(self, session_builder: async_sessionmaker[AsyncSession], redis_client: Redis,
                            request: GetForecastRequest,
                            broker_producer: BaseBrokerProducer) -> GetForecastResponse | None:
        request_start = time.perf_counter()

        key_raw = f'{request.isin}#{request.time_frame}#{request.forecast_period}#{request.provide_plot}'
        key = hashlib.sha256(key_raw.encode()).hexdigest()

        cached_forecast = await redis_client.get(key)

        response = None

        if cached_forecast is not None:
            cached_response = CachedForecastResponse.model_validate_json(cached_forecast)

            forecast_request = ForecastRequest(isin=cached_response.isin, time_frame=cached_response.time_frame,
                                               requested_plot=cached_response.provide_plot, model=cached_response.model,
                                               status=ForecastRequestStatus.COMPLETED, used_cache=True,
                                               duration_ms=time.perf_counter() - request_start)

            await self._forecast_repository.save_request(session_builder, forecast_request)

            response = GetForecastResponse.model_validate(cached_response, from_attributes=True)
        else:
            forecast_request = ForecastRequest(isin=request.isin, time_frame=request.time_frame,
                                               requested_plot=request.provide_plot,
                                               status=ForecastRequestStatus.PENDING, used_cache=False)

            await self._forecast_repository.save_request(session_builder, forecast_request)

            await redis_client.setex(forecast_request.id, 600, request_start)

            await broker_producer.send(request)

        return response

    async def clear_history(self, session_builder: async_sessionmaker[AsyncSession]):
        await self._forecast_repository.clear_requests_history(session_builder)
