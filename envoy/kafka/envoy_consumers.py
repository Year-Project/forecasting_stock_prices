from aiokafka import AIOKafkaConsumer
from aiogram import Bot

from schemas.broker_messages.admin_secret_message import AdminSecretMessage
from schemas.broker_messages.forecast_publish_message import ForecastPublishMessage


async def start_forecast_consumer(bot: Bot, bootstrap: str, topic: str):
    consumer = AIOKafkaConsumer(
        topic,
        bootstrap_servers=bootstrap,
        group_id="envoy_forecast_publish_consumer_group",
    )

    await consumer.start()
    try:
        async for msg in consumer:
            payload = ForecastPublishMessage.model_validate_json(msg.value)
            await bot.send_message(payload.telegram_id, f"📈 Прогноз для {payload.isin}\n"
                                                        f"(timeframe = {payload.time_frame} |"
                                                        f" горизонт прогноза = {payload.forecast_period})\n"
                                                        f"Цена: {payload.forecast_price}\n")
    finally:
        await consumer.stop()


async def start_admin_secret_consumer(bot: Bot, bootstrap: str, topic: str):
    consumer = AIOKafkaConsumer(
        topic,
        bootstrap_servers=bootstrap,
        group_id="envoy_admin_secret_consumer_group",
    )

    await consumer.start()
    try:
        async for msg in consumer:
            payload = AdminSecretMessage.model_validate_json(msg.value)
            await bot.send_message(payload.telegram_id, f"🔐 Admin secret:\n`{payload.secret}`",
                                   parse_mode="Markdown")
    finally:
        await consumer.stop()
