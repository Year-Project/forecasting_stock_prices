import asyncio
import logging
import time

from aiokafka import AIOKafkaConsumer
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession

from postman.src.repositories.forecast_requests_repository import ForecastRequestsRepository
from postman.src.schemas.shared.cached_forecast_response import CachedForecastResponse
from postman.src.schemas.shared.forecast_request_status import ForecastRequestStatus
from postman.src.schemas.shared.forecast_response_message import ForecastResponseMessage
from postman.src.services.cache_utils import build_cache_key, build_request_start_key

logger = logging.getLogger(__name__)


async def consume_forecast_responses(
    consumer: AIOKafkaConsumer,
    session_builder: async_sessionmaker[AsyncSession],
    redis_client: Redis,
    forecast_repository: ForecastRequestsRepository,
) -> None:
    try:
        async for msg in consumer:
            try:
                payload = ForecastResponseMessage.model_validate_json(msg.value.decode("utf-8"))
            except Exception:
                logger.exception("Failed to parse forecast response message")
                continue

            duration_ms = None
            start_key = build_request_start_key(payload.request_id)
            started = await redis_client.get(start_key)
            if started is not None:
                try:
                    duration_ms = time.perf_counter() - float(started)
                except ValueError:
                    duration_ms = None
                await redis_client.delete(start_key)

            if payload.status == ForecastRequestStatus.COMPLETED.value:
                if payload.forecast_price is not None:
                    cached_response = CachedForecastResponse(
                        isin=payload.isin,
                        forecast_period=payload.forecast_period,
                        time_frame=payload.time_frame,
                        forecast_price=payload.forecast_price,
                        forecast_confidence=payload.forecast_confidence,
                        forecast_plot=payload.forecast_plot,
                        provide_plot=payload.provide_plot,
                        model=payload.model or "auto_arima",
                    )
                    cache_key = build_cache_key(
                        payload.isin,
                        payload.time_frame,
                        payload.forecast_period,
                        payload.provide_plot,
                    )
                    await redis_client.setex(cache_key, 600, cached_response.model_dump_json())

                await forecast_repository.update_request_status(
                    session_builder,
                    payload.request_id,
                    status=ForecastRequestStatus.COMPLETED,
                    error=None,
                    model=payload.model,
                    duration_ms=duration_ms,
                )
            else:
                await forecast_repository.update_request_status(
                    session_builder,
                    payload.request_id,
                    status=ForecastRequestStatus.FAILED,
                    error=payload.error or "Forecast failed",
                    model=payload.model,
                    duration_ms=duration_ms,
                )
    except asyncio.CancelledError:
        return
