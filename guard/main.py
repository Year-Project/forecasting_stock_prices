import logging.config
import os
from contextlib import asynccontextmanager
from pathlib import Path

from aiokafka import AIOKafkaProducer
from dotenv import load_dotenv
from fastapi import FastAPI
import uvicorn

from dependencies.dependencies import db_registry
from guard.src.dependencies import set_admin_secret_producer
from guard.src.handlers.v1 import auth_handler
from guard.src.handlers.v1 import internal_admin_handler
from exceptions.exception_handler import service_exception_handler, BaseServiceException
from guard.src.kafka.admin_secret_producer import AdminSecretProducer
from utils.config_utils import load_yaml_config, ROOT_DIR

load_dotenv(Path(ROOT_DIR / ".env"))
load_dotenv()

LOGGER_CONFIG_PATH = Path(ROOT_DIR / "logger/config")

KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS")
KAFKA_SECURITY_PROTOCOL = os.getenv("KAFKA_SECURITY_PROTOCOL")
KAFKA_ADMIN_SECRET_TOPIC = os.getenv("KAFKA_ADMIN_SECRET_TOPIC")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # app startup
    logging.config.dictConfig(
        load_yaml_config(
            f"{LOGGER_CONFIG_PATH}/logger_conf.{os.getenv('ENV', 'testing')}.yaml"
        )
    )
    db_registry.register("auth", os.getenv("DATABASE_URL"))

    producer = AIOKafkaProducer(bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS, security_protocol=KAFKA_SECURITY_PROTOCOL)
    await producer.start()

    admin_secret_producer = AdminSecretProducer(producer, KAFKA_ADMIN_SECRET_TOPIC)
    set_admin_secret_producer(admin_secret_producer)

    try:
        yield
    finally:
        # app shutdown
        await producer.stop()


app = FastAPI(title="Guard Service", description="Authentication and authorization service", lifespan=lifespan)

app.include_router(auth_handler.router, prefix="/guard")
app.include_router(internal_admin_handler.router, prefix="/guard")
app.add_exception_handler(BaseServiceException, service_exception_handler)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)

