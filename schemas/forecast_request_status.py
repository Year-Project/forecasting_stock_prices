from enum import Enum


class ForecastRequestStatus(str, Enum):
    COMPLETED = 'completed'
    PENDING = 'pending'
    FAILED = 'failed'
