import json
from typing import override

from aiokafka import AIOKafkaProducer

from Postman.src.schemas.Request.GetForecastRequest import GetForecastRequest
from kafka.BaseBrokerProducer import BaseBrokerProducer


class ForecastRequestProducer(BaseBrokerProducer[GetForecastRequest]):
    def __init__(self, producer: AIOKafkaProducer, topic: str):
        self._producer = producer
        self._topic = topic

    @override
    async def send(self, payload: GetForecastRequest) -> None:
        await self._producer.send_and_wait(self._topic,
                                           json.dumps(payload.model_dump(mode="json"), default=str).encode("utf-8"))