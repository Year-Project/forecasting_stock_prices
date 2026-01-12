from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from magician.services.forecast_service import ForecastService
    from magician.services.scavenger_client import ScavengerClient

__all__ = ["ForecastService", "ScavengerClient"]
