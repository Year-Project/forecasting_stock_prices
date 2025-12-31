import logging
from datetime import datetime
from uuid import UUID

from sqlalchemy import select, Select, text, update, Update
from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession

from postman.src.models.forecast_request import ForecastRequest
from postman.src.schemas.request.get_forecasts_info_request import GetForecastsInfoRequest
from schemas.forecast_request_status import ForecastRequestStatus


class ForecastRequestsRepository:
    @staticmethod
    async def _build_select_requests_query(isin: str | None = None, time_frame: str | None = None,
                                           requested_plot: bool | None = None,
                                           model: str | None = None, user_id: UUID | None = None,
                                           error: str | None = None, used_cache: bool | None = None,
                                           from_date: datetime | None = None,
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
    async def _build_update_requests_query(request_id: UUID, status: ForecastRequestStatus,
                                           model: str | None = None, error: str | None = None,
                                           duration_ms: float | None = None) -> Update:
        fields = {
            "status": status,
            "error": error,
            "model": model,
            "duration_ms": duration_ms,
        }

        active_fields = {field_name: field_value for field_name, field_value in
                         fields.items() if field_value is not None}

        query = update(ForecastRequest).where(ForecastRequest.id == request_id).values(**active_fields)

        return query

    @staticmethod
    async def save_request(session_builder: async_sessionmaker[AsyncSession], forecast_request: ForecastRequest):
        async with session_builder() as session:
            async with session.begin():
                session.add(forecast_request)

    @staticmethod
    async def get_request_by_id(session_builder: async_sessionmaker[AsyncSession],
                                request_id: UUID) -> ForecastRequest | None:
        async with session_builder() as session:
            query = select(ForecastRequest).where(ForecastRequest.id == request_id)

            result = await session.execute(query)

            return result.scalar_one_or_none()

    async def select_requests(self, session_builder: async_sessionmaker[AsyncSession],
                              stats_request: GetForecastsInfoRequest) -> list[ForecastRequest]:
        async with session_builder() as session:
            query = await self._build_select_requests_query(stats_request.isin,
                                                            stats_request.time_frame,
                                                            stats_request.requested_plot, stats_request.model,
                                                            stats_request.user_id, stats_request.error,
                                                            stats_request.used_cache, stats_request.from_date,
                                                            stats_request.to_date)

            result = await session.execute(query)

            return list(result.scalars().all())

    async def update_forecast_request(self, session_builder: async_sessionmaker[AsyncSession], request_id: UUID,
                                      status: ForecastRequestStatus, model: str | None = None, error: str | None = None,
                                      duration_ms: float | None = None):
        async with session_builder() as session:
            async with session.begin():
                query = await self._build_update_requests_query(request_id, status, model, error, duration_ms)

                await session.execute(query)

    @staticmethod
    async def clear_requests_history(session_builder: async_sessionmaker[AsyncSession]):
        async with session_builder() as session:
            async with session.begin():
                await session.execute(text(f"TRUNCATE TABLE forecast_requests"))
