from uuid import uuid4

from postman.src.schemas.shared.cached_forecast_response import CachedForecastResponse
from schemas.broker_messages.forecast_publish_message import ForecastPublishMessage
from schemas.broker_messages.forecast_response_message import ForecastResponseMessage
from schemas.forecast_request_status import ForecastRequestStatus


def test_forecast_return_survives_response_cache_and_publish_schema():
    message = ForecastResponseMessage(
        request_id=uuid4(),
        isin="SBER",
        forecast_period=7,
        time_frame="1d",
        forecast_price=110.0,
        forecast_return=0.1,
        provide_plot=False,
        model="models:/stock_return_forecaster_week@prd:ridge:SBER",
        status=ForecastRequestStatus.COMPLETED,
    )

    cached = CachedForecastResponse.model_validate(message, from_attributes=True)
    published = ForecastPublishMessage.model_validate(message, from_attributes=True)

    assert cached.forecast_return == 0.1
    assert published.forecast_return == 0.1
