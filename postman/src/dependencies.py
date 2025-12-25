from aiokafka import AIOKafkaProducer

from postman.src.kafka.forecast_request_producer import ForecastRequestProducer
from db.registry import DatabaseRegistry

db_registry = DatabaseRegistry()
forecast_request_producer: ForecastRequestProducer | None = None


def get_session_maker(name: str):
    return db_registry.get(name).sessionmaker


def get_forecast_requests_session_maker():
    return get_session_maker("forecast_requests")


def get_forecast_request_producer() -> ForecastRequestProducer | None:
    return forecast_request_producer


def set_forecast_request_producer(producer: ForecastRequestProducer):
    global forecast_request_producer
    forecast_request_producer = producer

