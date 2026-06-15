import logging

from aiogram import Router
from aiogram.fsm.context import FSMContext
from aiogram.types import Message
from aiogram.filters import Command

from aiogram.fsm.state import StatesGroup, State
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
from aiogram.types import CallbackQuery


class ForecastFSM(StatesGroup):
    waiting_isin = State()
    waiting_period = State()
    waiting_plot = State()


def period_keyboard():
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="5d", callback_data="horizon:5d"),
                InlineKeyboardButton(text="21d", callback_data="horizon:21d"),
            ]
        ]
    )


def plot_keyboard():
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="📈 Да", callback_data="plot:yes"),
                InlineKeyboardButton(text="❌ Нет", callback_data="plot:no"),
            ]
        ]
    )


router = Router()


def setup(forecast_client):
    @router.message(Command("forecast"))
    async def forecast_start(message: Message, state: FSMContext):
        await state.clear()
        await message.answer("📌 Введите ISIN инструмента:")
        await state.set_state(ForecastFSM.waiting_isin)

    @router.message(ForecastFSM.waiting_isin)
    async def forecast_isin(message: Message, state: FSMContext):
        isin = message.text.strip().upper()

        await state.update_data(isin=isin)
        await message.answer("📆 Выберите период прогнозирования:", reply_markup=period_keyboard())
        await state.set_state(ForecastFSM.waiting_period)

    @router.callback_query(ForecastFSM.waiting_period, lambda c: c.data.startswith("horizon:"))
    async def forecast_period(callback: CallbackQuery, state: FSMContext):
        period = int(callback.data.split(":")[1].replace('d', ''))
        await state.update_data(forecast_period=period)
        await state.set_state(ForecastFSM.waiting_plot)
        await callback.answer()
        await callback.message.answer("📈 Нужен график?", reply_markup=plot_keyboard())

    @router.callback_query(ForecastFSM.waiting_plot, lambda c: c.data.startswith("plot:"))
    async def forecast_plot(callback: CallbackQuery, state: FSMContext):
        provide_plot = callback.data.endswith("yes")

        data = await state.get_data()
        await state.clear()

        request = {
            "isin": data["isin"],
            "forecast_period": data["forecast_period"],
            "time_frame": '1d',
            "provide_plot": provide_plot,
        }

        response = await forecast_client.request_forecast(telegram_id=callback.from_user.id,
                                                          payload=request)
        response_body = response.json()

        if response.status_code == 200:
            forecast_return = response_body.get("forecast_return")
            return_line = f"Доходность: {forecast_return}\n" if forecast_return is not None else ""
            await callback.message.answer(f"📈 Прогноз для {response_body['isin']}\n"
                                          f"(timeframe = {response_body['time_frame']} |"
                                          f" горизонт прогноза = {response_body['forecast_period']})\n"
                                          f"Цена: {response_body['forecast_price']}\n"
                                          f"{return_line}")
        elif response.status_code == 201:
            await callback.message.answer("⏳ Прогноз запрошен. Я пришлю результат, когда он будет готов.")
        else:
            await callback.message.answer("❌ Ошибка при запросе прогноза.")

        await callback.answer()

    return router
