from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession

from guard.src.models.refresh_token import RefreshToken


class RefreshTokenRepository:
    @staticmethod
    async def save_token(session_builder: async_sessionmaker[AsyncSession], refresh_token: RefreshToken):
        async with session_builder() as session:
            async with session.begin():
                session.add(refresh_token)

    @staticmethod
    async def get_token_by_id(session_builder: async_sessionmaker[AsyncSession],
                              token_id: UUID) -> RefreshToken | None:
        async with session_builder() as session:
            query = select(RefreshToken).where(RefreshToken.id == token_id)

            result = await session.execute(query)

            return result.scalar_one_or_none()

    @staticmethod
    async def revoke_token(session_builder: async_sessionmaker[AsyncSession], token_id: UUID):
        async with session_builder() as session:
            async with session.begin():
                query = update(RefreshToken).where(
                    RefreshToken.id == token_id).values(revoked_at=datetime.now(timezone.utc))

                await session.execute(query)

    @staticmethod
    async def revoke_all_user_tokens(session_builder: async_sessionmaker[AsyncSession], user_id: UUID):
        async with session_builder() as session:
            async with session.begin():
                query = update(RefreshToken).where(
                    RefreshToken.user_id == user_id, RefreshToken.revoked_at.is_(None)).values(
                    revoked_at=datetime.now(timezone.utc))

                await session.execute(query)

    @staticmethod
    async def is_token_revoked(session_builder: async_sessionmaker[AsyncSession], token_id: UUID) -> bool:
        token = await RefreshTokenRepository.get_token_by_id(session_builder, token_id)

        if token is None:
            return True

        return token.revoked_at is not None

