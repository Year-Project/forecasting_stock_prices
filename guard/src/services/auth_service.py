import uuid
from datetime import datetime, timezone
from typing import Annotated
from uuid import UUID

from fastapi import Depends
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession

from guard.src.models.refresh_token import RefreshToken
from guard.src.models.user import User
from guard.src.repositories.refresh_token_repository import RefreshTokenRepository
from guard.src.repositories.user_repository import UserRepository
from guard.src.schemas.request.admin_auth_request import AdminAuthRequest
from guard.src.schemas.request.auth_request import AuthRequest
from guard.src.schemas.response.admin_auth_response import AdminAuthResponse
from guard.src.schemas.response.auth_response import AuthResponse
from schemas.user_role import UserRole
from utils.hash_utils import hash_entity
from guard.src.exceptions.exceptions import InvalidCredentialsException, TokenRevokedException, UserNotFound
from utils.jwt_utils import create_access_token, create_refresh_token, get_token_expiration, decode_token, \
    verify_and_decode_token

import json


class AuthService:
    def __init__(self, user_repository: Annotated[UserRepository, Depends()],
                 refresh_token_repository: Annotated[RefreshTokenRepository, Depends()]):
        self._user_repository = user_repository
        self._refresh_token_repository = refresh_token_repository

    async def _revoke_token(self, session_builder: async_sessionmaker[AsyncSession],
                            redis_client: Redis, token_id: UUID):
        cache_key = f"guard:refresh_token:{token_id}"
        await redis_client.delete(cache_key)
        await self._refresh_token_repository.revoke_token(session_builder, token_id)

    async def _revoke_tokens_for_user(self, session_builder: async_sessionmaker[AsyncSession],
                                      redis_client: Redis, user_id: UUID):
        cache_key = f"guard:user_token_version:{user_id}"
        await redis_client.incr(cache_key)
        await self._refresh_token_repository.revoke_all_user_tokens(session_builder, user_id)

    @staticmethod
    async def _cache_token_info(redis_client: Redis, refresh_token: str, token_id: UUID, user: User, version: int):
        refresh_payload = decode_token(refresh_token)

        if refresh_payload is not None:
            refresh_exp = refresh_payload.get("exp", 0)
            refresh_ttl = max(0, refresh_exp - int(datetime.now(timezone.utc).timestamp()))
            cache_key = f"guard:refresh_token:{token_id}"
            cache_data = {
                "user_id": str(user.id),
                "version": version,
            }
            await redis_client.setex(cache_key, refresh_ttl, json.dumps(cache_data))

    @staticmethod
    async def _get_refresh_tokens_version(redis_client: Redis, user_id: UUID) -> int:
        cache_key = f"guard:user_token_version:{user_id}"
        refresh_token_version = redis_client.get(cache_key)

        if refresh_token_version is None:
            refresh_token_version = 1
            await redis_client.set(cache_key, 1)

        return refresh_token_version

    async def _create_tokens(self, session_builder: async_sessionmaker[AsyncSession], redis_client: Redis,
                             user: User, refresh_token_version: int) -> dict[str, str]:
        access_token = create_access_token(user.id, user.role)

        token_id = uuid.uuid4()
        refresh_token = create_refresh_token(token_id, user.id, refresh_token_version)

        token_hash = hash_entity(refresh_token)
        refresh_token_orm = RefreshToken(id=token_id, user_id=user.id, token_hash=token_hash,
                                         expires_at=get_token_expiration(refresh_token))

        await self._refresh_token_repository.save_token(session_builder, refresh_token_orm)

        await self._cache_token_info(redis_client, refresh_token, refresh_token_orm.id, user, refresh_token_version)

        return {"access_token": access_token, "refresh_token": refresh_token}

    async def authenticate(self, session_builder: async_sessionmaker[AsyncSession],
                           redis_client: Redis, request: AuthRequest) -> AuthResponse:
        user = await self._user_repository.get_or_create_user(session_builder, request.telegram_id)

        refresh_token_version = await self._get_refresh_tokens_version(redis_client, user.id)

        tokens = await self._create_tokens(session_builder, redis_client, user, refresh_token_version)

        return AuthResponse(access_token=tokens["access_token"], refresh_token=tokens["refresh_token"])

    async def refresh_tokens(self, session_builder: async_sessionmaker[AsyncSession],
                             redis_client: Redis, refresh_token: str) -> AuthResponse:
        payload = verify_and_decode_token(refresh_token, "refresh")

        user_id = UUID(payload["sub"])
        refresh_token_version = payload["version"]
        token_id = payload["jti"]

        valid_refresh_token_version = await self._get_refresh_tokens_version(redis_client, user_id)

        if refresh_token_version != valid_refresh_token_version:
            raise TokenRevokedException()

        cached_token = await redis_client.get(token_id)

        if cached_token is None:
            raise TokenRevokedException()

        user = await self._user_repository.get_by_id(session_builder, user_id)

        if user is None:
            raise InvalidCredentialsException("User referenced in jwt payload not found",
                                              meta={'user_id': payload["sub"]})

        await self._revoke_token(session_builder, redis_client, token_id)

        tokens = await self._create_tokens(session_builder, redis_client, user, refresh_token_version)

        return AuthResponse(access_token=tokens["access_token"], refresh_token=tokens["refresh_token"])

    async def get_admin_token(self, session_builder: async_sessionmaker[AsyncSession],
                              request: AdminAuthRequest) -> AdminAuthResponse:
        user = await self._user_repository.get_by_telegram_id(session_builder, request.telegram_id)

        if user is None:
            raise UserNotFound(meta={'provided telegram_id': request.telegram_id})

        if user.role != UserRole.ADMIN:
            raise InvalidCredentialsException('Requested admin token for user without admin permissions',
                                              meta={'provided telegram_id': request.telegram_id})

        if user.admin_secret != hash_entity(request.secret_key):
            raise InvalidCredentialsException('Provided invalid admin secret',
                                              meta={'provided telegram_id': request.telegram_id,
                                                    'provided secret': request.secret_key})

        access_token = create_access_token(user.id, UserRole.ADMIN)

        return AdminAuthResponse(access_token=access_token)
