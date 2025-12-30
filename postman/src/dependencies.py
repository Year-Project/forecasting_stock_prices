from fastapi import Depends

from dependencies.dependencies import get_session_maker
from postman.src.kafka.forecast_request_producer import ForecastRequestProducer
from postman.src.kafka.forecast_publish_producer import ForecastPublishProducer
from postman.src.kafka.forecast_response_consumer import ForecastResponseConsumer
from postman.src.repositories.forecast_requests_repository import ForecastRequestsRepository
from postman.src.services.forecast_service import ForecastService

forecast_request_producer: ForecastRequestProducer | None = None
forecast_publish_producer: ForecastPublishProducer | None = None
forecast_response_consumer: ForecastResponseConsumer | None = None


def get_forecast_requests_session_maker():
    return get_session_maker("forecast_requests")


def get_forecast_request_producer() -> ForecastRequestProducer | None:
    return forecast_request_producer


def get_forecast_service(repository: ForecastRequestsRepository = Depends()) -> ForecastService:
    return ForecastService(repository)


def set_forecast_request_producer(producer: ForecastRequestProducer):
    global forecast_request_producer
    forecast_request_producer = producer


def get_forecast_publish_producer() -> ForecastPublishProducer | None:
    return forecast_publish_producer


def set_forecast_publish_producer(producer: ForecastPublishProducer):
    global forecast_publish_producer
    forecast_publish_producer = producer


def get_forecast_response_consumer() -> ForecastResponseConsumer | None:
    return forecast_response_consumer


def set_forecast_response_consumer(consumer: ForecastResponseConsumer):
    global forecast_response_consumer
    forecast_response_consumer = consumer

