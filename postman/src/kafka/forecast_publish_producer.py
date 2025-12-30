import json
from typing import override

from aiokafka import AIOKafkaProducer

from kafka.BaseBrokerProducer import BaseBrokerProducer
from schemas.broker_messages.forecast_prediction_message import ForecastPredictionMessage


class ForecastPublishProducer(BaseBrokerProducer[ForecastPredictionMessage]):
    def __init__(self, producer: AIOKafkaProducer, topic: str):
        self._producer = producer
        self._topic = topic

    @override
    async def send(self, payload: ForecastPredictionMessage) -> None:
        await self._producer.send_and_wait(self._topic,
                                           json.dumps(payload.model_dump(mode="json")).encode("utf-8"))



