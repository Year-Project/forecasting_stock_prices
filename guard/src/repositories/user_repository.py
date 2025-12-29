from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession

from guard.src.models.user import User
from schemas.user_role import UserRole


class UserRepository:
    @staticmethod
    async def get_by_telegram_id(session_builder: async_sessionmaker[AsyncSession], telegram_id: int) -> User | None:
        async with session_builder() as session:
            query = select(User).where(User.telegram_id == telegram_id)

            result = await session.execute(query)

            return result.scalar_one_or_none()

    @staticmethod
    async def get_by_id(session_builder: async_sessionmaker[AsyncSession], user_id: UUID) -> User | None:
        async with session_builder() as session:
            query = select(User).where(User.id == user_id)

            result = await session.execute(query)

            return result.scalar_one_or_none()

    @staticmethod
    async def create_user(session_builder: async_sessionmaker[AsyncSession], user: User):
        async with session_builder() as session:
            async with session.begin():
                session.add(user)

    @staticmethod
    async def get_or_create_user(session_builder: async_sessionmaker[AsyncSession], telegram_id: int,
                                 role: UserRole = UserRole.USER) -> User:
        user = await UserRepository.get_by_telegram_id(session_builder, telegram_id)
        if user is None:
            user = User(telegram_id=telegram_id, role=role)

            await UserRepository.create_user(session_builder, user)
        return user

    @staticmethod
    async def update_user_role(session_builder: async_sessionmaker[AsyncSession], user_id: UUID,
                               role: UserRole = UserRole.USER):
        async with session_builder() as session:
            async with session.begin():
                query = update(User).where(User.id == user_id).values(role=role)

                await session.execute(query)


