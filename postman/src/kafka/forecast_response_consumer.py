from typing import override

from aiokafka import AIOKafkaConsumer
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession

from kafka.BaseBrokerConsumer import BaseBrokerConsumer
from postman.src.kafka.forecast_publish_producer import ForecastPublishProducer
from postman.src.services.forecast_service import ForecastService
from schemas.broker_messages.forecast_response_message import ForecastResponseMessage


class ForecastResponseConsumer(BaseBrokerConsumer[ForecastResponseMessage]):
    def __init__(self, consumer: AIOKafkaConsumer, forecast_service: ForecastService,
                 session_builder: async_sessionmaker[AsyncSession], redis_client: Redis,
                 broker_producer: ForecastPublishProducer):
        self._consumer = consumer
        self._forecast_service = forecast_service
        self._session_builder = session_builder
        self._redis_client = redis_client
        self._broker_producer = broker_producer

    @override
    async def consume(self, payload: ForecastResponseMessage):
        await self._forecast_service.process_async_forecast(self._session_builder, self._redis_client,
                                                            self._broker_producer, payload)

    async def start(self):
        async for message in self._consumer:
            try:
                payload = ForecastResponseMessage.model_validate_json(message.value)
                await self.consume(payload)
            except ValueError as e:
                continue

