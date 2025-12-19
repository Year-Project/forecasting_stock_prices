from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession

from Postman.src.models.ForecastRequest import ForecastRequest


class ForecastRequestsRepository:
    @staticmethod
    async def save_request(session_builder: async_sessionmaker[AsyncSession], forecast_request: ForecastRequest):
        async with session_builder() as session:
            session.add(forecast_request)
            await session.commit()
