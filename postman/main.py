import logging.config
import os
from contextlib import asynccontextmanager
from pathlib import Path

from aiokafka import AIOKafkaProducer
from dotenv import load_dotenv
from fastapi import FastAPI
import uvicorn

from postman.src.kafka.forecast_request_producer import ForecastRequestProducer
from db.cache import redis_pool
from postman.src.dependencies import db_registry, set_forecast_request_producer
from postman.src.handlers.v1 import forecast_handler, requests_info_handler
from exceptions.exception_handler import service_exception_handler, BaseServiceException
from utils.utils import load_yaml_config, ROOT_DIR

load_dotenv(Path(ROOT_DIR / ".env"))
load_dotenv()

LOGGER_CONFIG_PATH = Path(ROOT_DIR / "logger/config")

KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS")
KAFKA_SECURITY_PROTOCOL = os.getenv("KAFKA_SECURITY_PROTOCOL")
KAFKA_FORECAST_REQUEST_TOPIC = os.getenv("KAFKA_FORECAST_REQUEST_TOPIC")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # app startup
    logging.config.dictConfig(
        load_yaml_config(
            f"{LOGGER_CONFIG_PATH}/logger_conf.{os.getenv('ENV', 'testing')}.yaml"
        )
    )
    db_registry.register("forecast_requests", os.getenv("DATABASE_URL"))

    producer = AIOKafkaProducer(bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS, security_protocol=KAFKA_SECURITY_PROTOCOL)
    await producer.start()

    forecast_request_producer = ForecastRequestProducer(producer, KAFKA_FORECAST_REQUEST_TOPIC)
    set_forecast_request_producer(forecast_request_producer)

    try:
        yield
    finally:
        # app shutdown
        await redis_pool.disconnect()
        await producer.stop()


app = FastAPI(title="Postman Service", description="API service to get forecasts for provided ISINs", lifespan=lifespan)

app.include_router(forecast_handler.router, prefix="/postman")
app.include_router(requests_info_handler.router, prefix="/postman")
app.add_exception_handler(BaseServiceException, service_exception_handler)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
