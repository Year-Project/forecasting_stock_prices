from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import update, select
from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession

from guard.src.models.admin_secret import AdminSecret


class AdminSecretRepository:

    @staticmethod
    async def get_secret_by_user_id(session_builder: async_sessionmaker[AsyncSession],
                                    user_id: UUID) -> AdminSecret | None:
        async with session_builder() as session:
            query = select(AdminSecret).where(AdminSecret.user_id == user_id, AdminSecret.revoked_at.is_(None))

            result = await session.execute(query)

            return result.scalar_one_or_none()

    @staticmethod
    async def save_secret(session_builder: async_sessionmaker[AsyncSession], secret: AdminSecret):
        async with session_builder() as session:
            async with session.begin():
                session.add(secret)

    @staticmethod
    async def revoke_admin_secret(session_builder: async_sessionmaker[AsyncSession], user_id: UUID):
        async with session_builder() as session:
            async with session.begin():
                query = update(AdminSecret).where(
                    AdminSecret.user_id == user_id, AdminSecret.revoked_at.is_(None)).values(
                    revoked_at=datetime.now(timezone.utc))

                await session.execute(query)

    @staticmethod
    async def is_admin_secret_revoked(session_builder: async_sessionmaker[AsyncSession], user_id: UUID) -> bool:
        secret = await AdminSecretRepository.get_secret_by_user_id(session_builder, user_id)

        if secret is None:
            return True

        return secret.revoked_at is not None
