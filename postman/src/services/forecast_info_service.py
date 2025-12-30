import os
import re
from pathlib import Path
from typing import Annotated

import numpy as np
from fastapi import Depends
from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession

from postman.src.repositories.forecast_requests_repository import ForecastRequestsRepository
from postman.src.schemas.request.get_forecasts_info_request import GetForecastsInfoRequest
from postman.src.schemas.response.get_forecasts_history_response import (
    ForecastsHistory,
    GetForecastsHistoryResponse,
)
from postman.src.schemas.response.get_forecasts_stats_response import GetForecastsStatsResponse
from utils.utils import load_yaml_config, ROOT_DIR


class ForecastInfoService:
    CONFIG_PATH = Path(ROOT_DIR / 'postman/src/services/config')

    def __init__(self, forecast_repository: Annotated[ForecastRequestsRepository, Depends()]):
        parsed_yaml = load_yaml_config(f"{self.CONFIG_PATH}/stats_metrics_conf.{os.getenv('ENV')}.yaml")
        self.metrics: list[str] = parsed_yaml['metrics']
        self._forecast_repository = forecast_repository

    @staticmethod
    def get_mean_duration(durations: list[float]) -> float:
        return float(np.mean(durations))

    @staticmethod
    def get_quantile_duration(durations: list[float], quantile: float) -> float:
        return float(np.quantile(durations, quantile))

    async def get_stats(self, session_builder: async_sessionmaker[AsyncSession],
                        request: GetForecastsInfoRequest) -> GetForecastsStatsResponse:

        collected_requests = await self._forecast_repository.select_requests(session_builder, request)

        response = GetForecastsStatsResponse.model_validate(request, from_attributes=True)

        durations = [forecast_request.duration_ms for forecast_request in collected_requests
                     if forecast_request.duration_ms is not None]

        duration_stats: dict[str, float | None] = {metric: None for metric in self.metrics}

        if durations:
            for metric in self.metrics:
                if metric == 'mean':
                    duration_stats[metric] = self.get_mean_duration(durations)
                elif metric.startswith('quantile'):
                    match = re.search(r'(\d+)%', metric)

                    if match:
                        quantile_value = float(match.group(1)) / 100
                        duration_stats[metric] = self.get_quantile_duration(durations, quantile_value)

        response.execution_duration = duration_stats

        return response

    async def get_history(self, session_builder: async_sessionmaker[AsyncSession],
                          request: GetForecastsInfoRequest) -> GetForecastsHistoryResponse:
        collected_requests = await self._forecast_repository.select_requests(session_builder, request)

        history = [
            ForecastsHistory.model_validate(item, from_attributes=True) for item in collected_requests
        ]

        return GetForecastsHistoryResponse(**request.model_dump(mode="json"), history=history)
