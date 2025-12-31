from aiogram import Router
from aiogram.types import Message
from aiogram.filters import Command

router = Router()


@router.message(Command("me"))
async def me_handler(message: Message):
    await message.answer(
        f"👤 Username: @{message.from_user.username}\n"
        f"🆔 Telegram ID: {message.from_user.id}"
    )
