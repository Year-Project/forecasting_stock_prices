import os
from datetime import datetime
from uuid import UUID

from sqlalchemy import select, Select, text
from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession

from postman.src.models.forecast_request import ForecastRequest
from postman.src.schemas.request.get_forecasts_info_request import GetForecastsInfoRequest


class ForecastRequestsRepository:
    @staticmethod
    async def _build_select_requests_query(isin: str | None = None, time_frame: str | None = None,
                                           requested_plot: bool | None = None, model: str | None = None,
                                           user_id: UUID | None = None, error: str | None = None,
                                           used_cache: bool | None = None, from_date: datetime | None = None,
                                           to_date: datetime | None = None) -> Select[tuple[ForecastRequest]]:

        query = select(ForecastRequest)

        filters = {
            "isin": isin,
            "time_frame": time_frame,
            "requested_plot": requested_plot,
            "model": model,
            "user_id": user_id,
            "error": error,
            "used_cache": used_cache,
        }

        active_filters = {filter_name: filter_value for filter_name, filter_value in
                          filters.items() if filter_value is not None}
        query = query.filter_by(**active_filters)

        if from_date is not None:
            query = query.where(ForecastRequest.created_at >= from_date)

        if to_date is not None:
            query = query.where(ForecastRequest.created_at <= to_date)

        return query

    @staticmethod
    async def save_request(session_builder: async_sessionmaker[AsyncSession], forecast_request: ForecastRequest):
        async with session_builder() as session:
            async with session.begin():
                session.add(forecast_request)

    async def select_requests(self, session_builder: async_sessionmaker[AsyncSession],
                              stats_request: GetForecastsInfoRequest) -> list[ForecastRequest]:
        async with session_builder() as session:
            query = await self._build_select_requests_query(stats_request.isin, stats_request.time_frame,
                                                            stats_request.requested_plot, stats_request.model,
                                                            stats_request.user_id, stats_request.error,
                                                            stats_request.used_cache, stats_request.from_date,
                                                            stats_request.to_date)

            result = await session.execute(query)

            return list(result.scalars().all())

    @staticmethod
    async def clear_requests_history(session_builder: async_sessionmaker[AsyncSession]):
        async with session_builder() as session:
            async with session.begin():
                await session.execute(text(f"TRUNCATE TABLE forecast_requests"))
