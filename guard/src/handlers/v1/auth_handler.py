import os
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession

from db.redis.cache import RedisClientProvider
from dependencies.dependencies import get_admin_user
from guard.src.dependencies import get_auth_session_maker
from guard.src.schemas.request.admin_auth_request import AdminAuthRequest
from guard.src.schemas.request.auth_request import AuthRequest
from guard.src.schemas.request.refresh_request import RefreshRequest
from guard.src.schemas.response.auth_response import AuthResponse
from guard.src.schemas.response.admin_auth_response import AdminAuthResponse
from guard.src.services.auth_service import AuthService
from schemas.current_user import CurrentUser

router = APIRouter(prefix="/auth/v1", tags=["auth"])

get_redis_client = RedisClientProvider(os.getenv('REDIS_USER_GUARD'), os.getenv('REDIS_PASSWORD_GUARD'))


@router.post("/auth", response_model=AuthResponse)
async def auth_handler(request: AuthRequest, auth_service: Annotated[AuthService, Depends()],
                       session_builder: async_sessionmaker[AsyncSession] = Depends(get_auth_session_maker),
                       redis_client: Redis = Depends(get_redis_client)):
    return await auth_service.authenticate(session_builder, redis_client, request)


@router.post("/refresh", response_model=AuthResponse)
async def refresh_handler(request: RefreshRequest, auth_service: Annotated[AuthService, Depends()],
                          session_builder: async_sessionmaker[AsyncSession] = Depends(get_auth_session_maker),
                          redis_client: Redis = Depends(get_redis_client)):
    return await auth_service.refresh_tokens(session_builder, redis_client, request.refresh_token)


@router.post("/auth-admin", response_model=AdminAuthResponse)
async def admin_token_handler(request: AdminAuthRequest,
                              auth_service: Annotated[AuthService, Depends()],
                              session_builder: async_sessionmaker[AsyncSession] = Depends(get_auth_session_maker)):
    return await auth_service.get_admin_token(session_builder, request)


@router.delete("/token_revoke/{user_id}")
async def token_revoke_handler(user_id: UUID, auth_service: Annotated[AuthService, Depends()],
                               admin: CurrentUser = Depends(get_admin_user),
                               session_builder: async_sessionmaker[AsyncSession] = Depends(get_auth_session_maker),
                               redis_client: Redis = Depends(get_redis_client)):
    return await auth_service.revoke_tokens_for_user(session_builder, redis_client, user_id)
