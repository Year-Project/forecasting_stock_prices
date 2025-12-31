import asyncio
import os

from aiogram import Bot, Dispatcher
from aiogram.types import BotCommand

from envoy.auth.auth_client import AuthClient
from envoy.utils.request_utils import ForecastClient
from envoy.handlers import common, forecast
from envoy.kafka.envoy_consumers import start_forecast_consumer, start_admin_secret_consumer

COMMANDS = [
    BotCommand(command="start", description="Запустить бота"),
    BotCommand(command="forecast", description="Получить прогноз по инструменту"),
    BotCommand(command="me", description="Показать информацию о пользователе"),
]


async def main():
    bot = Bot(os.getenv("BOT_TOKEN"))
    dp = Dispatcher()

    await bot.set_my_commands(COMMANDS)

    auth_client = AuthClient(os.getenv("AUTH_BASE_URL"))
    forecast_client = ForecastClient(os.getenv("FORECAST_BASE_URL"), auth_client)

    dp.include_router(common.router)
    dp.include_router(forecast.setup(forecast_client))

    await asyncio.gather(
        dp.start_polling(bot),
        start_forecast_consumer(bot, os.getenv("KAFKA_BOOTSTRAP_SERVERS"), os.getenv("KAFKA_FORECAST_PUBLISH_TOPIC")),
        start_admin_secret_consumer(bot, os.getenv("KAFKA_BOOTSTRAP_SERVERS"), os.getenv("KAFKA_ADMIN_SECRET_TOPIC")),
    )


if __name__ == "__main__":
    asyncio.run(main())
