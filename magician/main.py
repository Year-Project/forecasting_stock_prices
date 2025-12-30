import asyncio
import logging
import os
from contextlib import asynccontextmanager, suppress
from pathlib import Path

import httpx
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from dotenv import load_dotenv
from fastapi import FastAPI
import uvicorn

from magician.dependencies import set_scavenger_client
from magician.handlers import forecast_handler
from magician.schemas.request.get_forecast_request import GetForecastRequest
from magician.services.forecast_service import ForecastService
from magician.services.scavenger_client import ScavengerClient
from schemas.broker_messages.forecast_request_message import ForecastRequestMessage
from schemas.broker_messages.forecast_response_message import ForecastResponseMessage
from schemas.forecast_request_status import ForecastRequestStatus
from utils.config_utils import ROOT_DIR

load_dotenv(Path(ROOT_DIR / ".env"))
load_dotenv()

SCAVENGER_BASE_URL = os.getenv("SCAVENGER_BASE_URL", "http://localhost:8000")
SCAVENGER_TIMEOUT_SECS = float(os.getenv("SCAVENGER_TIMEOUT_SECS", "10"))
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS")
KAFKA_SECURITY_PROTOCOL = os.getenv("KAFKA_SECURITY_PROTOCOL", "PLAINTEXT")
KAFKA_FORECAST_REQUEST_TOPIC = os.getenv("KAFKA_FORECAST_REQUEST_TOPIC")
KAFKA_FORECAST_RESPONSE_TOPIC = os.getenv("KAFKA_FORECAST_RESPONSE_TOPIC")
MODEL_NAME = "auto_arima"

logger = logging.getLogger(__name__)


async def _consume_forecast_requests(
    consumer: AIOKafkaConsumer,
    producer: AIOKafkaProducer,
    forecast_service: ForecastService,
) -> None:
    async for msg in consumer:
        try:
            payload = ForecastRequestMessage.model_validate_json(msg.value.decode("utf-8"))
        except Exception:
            logger.exception("Failed to parse forecast request message")
            continue

        try:
            request = GetForecastRequest(
                isin=payload.isin,
                forecast_period=payload.forecast_period,
                time_frame=payload.time_frame,
                provide_plot=payload.provide_plot,
            )
            response = await forecast_service.get_forecast(request)
            response_message = ForecastResponseMessage(
                request_id=payload.request_id,
                isin=response.isin,
                forecast_period=response.forecast_period,
                time_frame=response.time_frame,
                forecast_price=response.forecast_price,
                forecast_confidence=response.forecast_confidence,
                forecast_plot=response.forecast_plot,
                provide_plot=payload.provide_plot,
                model=MODEL_NAME,
                status=ForecastRequestStatus.COMPLETED,
                error=None,
            )
        except Exception as exc:
            response_message = ForecastResponseMessage(
                request_id=payload.request_id,
                isin=payload.isin,
                forecast_period=payload.forecast_period,
                time_frame=payload.time_frame,
                provide_plot=payload.provide_plot,
                model=MODEL_NAME,
                status=ForecastRequestStatus.FAILED,
                error=str(exc),
            )

        await producer.send_and_wait(
            KAFKA_FORECAST_RESPONSE_TOPIC,
            response_message.model_dump_json().encode("utf-8"),
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    client = httpx.AsyncClient(base_url=SCAVENGER_BASE_URL, timeout=SCAVENGER_TIMEOUT_SECS)
    scavenger_client = ScavengerClient(client)
    set_scavenger_client(scavenger_client)
    forecast_service = ForecastService(scavenger_client)

    if not KAFKA_BOOTSTRAP_SERVERS or not KAFKA_FORECAST_REQUEST_TOPIC or not KAFKA_FORECAST_RESPONSE_TOPIC:
        raise RuntimeError("Kafka configuration is missing for Magician service.")

    producer = AIOKafkaProducer(bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS, security_protocol=KAFKA_SECURITY_PROTOCOL)
    consumer = AIOKafkaConsumer(
        KAFKA_FORECAST_REQUEST_TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        security_protocol=KAFKA_SECURITY_PROTOCOL,
        group_id="magician_forecast_requests_consumer_group"
    )
    await producer.start()
    await consumer.start()
    consumer_task = asyncio.create_task(_consume_forecast_requests(consumer, producer, forecast_service))
    try:
        yield
    finally:
        consumer_task.cancel()
        with suppress(asyncio.CancelledError):
            await consumer_task
        await consumer.stop()
        await producer.stop()
        await client.aclose()


app = FastAPI(
    title="Magician Service",
    description="Forecast worker that pulls history from Scavenger and returns forecasts.",
    lifespan=lifespan,
)

app.include_router(forecast_handler.router, prefix="/magician")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
