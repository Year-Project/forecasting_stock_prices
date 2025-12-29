import secrets
from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession

from guard.src.models.admin_secret import AdminSecret
from guard.src.models.user import User
from guard.src.repositories.admin_secret_repository import AdminSecretRepository
from guard.src.repositories.user_repository import UserRepository
from guard.src.schemas.request.admin_init_request import AdminInitRequest
from guard.src.schemas.response.admin_init_response import AdminInitResponse
from schemas.user_role import UserRole
from utils.hash_utils import hash_entity


class AdminInitService:
    def __init__(self, user_repository: Annotated[UserRepository, Depends()],
                 admin_secret_repository: Annotated[AdminSecretRepository, Depends()]):
        self._user_repository = user_repository
        self._admin_secret_repository = admin_secret_repository

    async def init_admin(self, session_builder: async_sessionmaker[AsyncSession],
                         request: AdminInitRequest) -> AdminInitResponse:
        user = await self._user_repository.get_by_telegram_id(session_builder, request.telegram_id)
        if user is None:
            await self._user_repository.create_user(session_builder,
                                                    User(telegram_id=request.telegram_id, role=UserRole.ADMIN))
            user = await self._user_repository.get_by_telegram_id(session_builder, request.telegram_id)
        elif user.role != UserRole.ADMIN:
            await self._user_repository.update_user_role(session_builder, user.id, UserRole.ADMIN)

        secret = secrets.token_urlsafe(32)
        secret_hash = hash_entity(secret)

        secret_orm = AdminSecret(user_id=user.id, secret_hash=secret_hash)

        await self._admin_secret_repository.save_secret(session_builder, secret_orm)

        return AdminInitResponse(user_id=user.id, telegram_id=user.telegram_id, secret=secret)
