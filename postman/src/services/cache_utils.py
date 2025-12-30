import hashlib
from uuid import UUID


def build_cache_key(isin: str, time_frame: str, forecast_period: int, provide_plot: bool) -> str:
    key_raw = f"{isin}#{time_frame}#{forecast_period}#{provide_plot}"
    return hashlib.sha256(key_raw.encode()).hexdigest()


def build_request_start_key(request_id: UUID) -> str:
    return f"request_start:{request_id}"
