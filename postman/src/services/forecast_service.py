import time
import hashlib

from typing import Annotated

from fastapi import Depends
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession

from postman.src.kafka.forecast_publish_producer import ForecastPublishProducer
from postman.src.kafka.forecast_request_producer import ForecastRequestProducer
from postman.src.models.forecast_request import ForecastRequest, ForecastRequestStatus
from postman.src.repositories.forecast_requests_repository import ForecastRequestsRepository
from postman.src.schemas.request.get_forecast_request import GetForecastRequest
from postman.src.schemas.response.get_forecast_response import GetForecastResponse
from postman.src.schemas.shared.cached_forecast_response import CachedForecastResponse
from schemas.broker_messages.forecast_prediction_message import ForecastPredictionMessage
from schemas.broker_messages.forecast_request_message import ForecastRequestMessage
from schemas.broker_messages.forecast_response_message import ForecastResponseMessage
from utils.parsing_utils import parse_time_frame, get_time_delta_from_time_frame


class ForecastService:
    def __init__(self, forecast_repository: ForecastRequestsRepository):
        self._forecast_repository = forecast_repository

    @staticmethod
    def _get_forecast_key(isin: str, time_frame_interval: int, time_frame_unit: str,
                          forecast_period: int, provide_plot) -> str:
        key_raw = f'{isin}#{time_frame_interval}#{time_frame_unit}#{forecast_period}#{provide_plot}'
        key = hashlib.sha256(key_raw.encode()).hexdigest()

        return f'postman:forecasts:{key}'

    async def get_forecasts(self, session_builder: async_sessionmaker[AsyncSession], redis_client: Redis,
                            request: GetForecastRequest,
                            broker_producer: ForecastRequestProducer) -> GetForecastResponse | None:
        request_start = time.perf_counter()

        time_frame_parsed = parse_time_frame(request.time_frame)

        cache_key = self._get_forecast_key(request.isin, time_frame_parsed.time_frame_interval,
                                           time_frame_parsed.time_frame_unit, request.forecast_period,
                                           request.provide_plot)

        cached_forecast = await redis_client.get(cache_key)

        response = None

        if cached_forecast is not None:
            cached_response = CachedForecastResponse.model_validate_json(cached_forecast)

            forecast_request = ForecastRequest(isin=cached_response.isin,
                                               time_frame_interval=cached_response.time_frame.time_frame_interval,
                                               time_frame_unit=cached_response.time_frame.time_frame_unit,
                                               requested_plot=cached_response.provide_plot, model=cached_response.model,
                                               status=ForecastRequestStatus.COMPLETED, used_cache=True,
                                               duration_ms=time.perf_counter() - request_start)

            await self._forecast_repository.save_request(session_builder, forecast_request)

            response = GetForecastResponse.model_validate(cached_response, from_attributes=True)
        else:
            forecast_request = ForecastRequest(isin=request.isin,
                                               time_frame_interval=time_frame_parsed.time_frame.time_frame_interval,
                                               time_frame_unit=time_frame_parsed.time_frame.time_frame_unit,
                                               requested_plot=request.provide_plot,
                                               status=ForecastRequestStatus.PENDING, used_cache=False)

            await self._forecast_repository.save_request(session_builder, forecast_request)

            await redis_client.setex(f"postman:duration:{forecast_request.id}", 600, request_start)

            broker_message = ForecastRequestMessage(request_id=forecast_request.id, isin=request.isin,
                                                    forecast_period=request.forecast_period,
                                                    time_frame=time_frame_parsed, provide_plot=request.provide_plot)

            await broker_producer.send(broker_message)

        return response

    async def clear_history(self, session_builder: async_sessionmaker[AsyncSession]):
        await self._forecast_repository.clear_requests_history(session_builder)

    async def process_async_forecast(self, session_builder: async_sessionmaker[AsyncSession],
                                     redis_client: Redis, broker_producer: ForecastPublishProducer,
                                     broker_message: ForecastResponseMessage):
        cache_key = self._get_forecast_key(broker_message.isin, broker_message.time_frame.time_frame_interval,
                                           broker_message.time_frame.time_frame_unit, broker_message.forecast_period,
                                           broker_message.provide_plot)

        cache_duration = get_time_delta_from_time_frame(broker_message.time_frame)

        cache_forecast = CachedForecastResponse.model_validate(broker_message, from_attributes=True)

        await redis_client.setex(cache_key, cache_duration, cache_forecast.model_dump_json())

        request_duration = redis_client.get(f"postman:duration:{broker_message.request_id}")

        await self._forecast_repository.update_forecast_request(session_builder, broker_message.request_id,
                                                                broker_message.status, broker_message.model,
                                                                broker_message.error, request_duration)

        forecast_publish_message = ForecastPredictionMessage.model_validate(broker_message, from_attributes=True)

        await broker_producer.send(forecast_publish_message)
