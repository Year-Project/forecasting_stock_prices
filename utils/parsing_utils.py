import datetime
from datetime import timedelta, datetime

from dateutil.relativedelta import relativedelta
import re

from schemas.parsed_timeframe import ParsedTimeframe


def parse_time_frame(timeframe: str) -> ParsedTimeframe | None:
    match = re.search(r'(\d+)(\w+)', timeframe)

    if match is None:
        return None

    return ParsedTimeframe(time_frame_interval=int(match.group(1)), timeframe_unit=match.group(2))


def get_time_delta_from_time_frame(time_frame: ParsedTimeframe) -> timedelta | None:
    units_map = {
        "s": "seconds",
        "m": "minutes",
        "h": "hours",
        "d": "days",
        "w": "weeks",
        "mo": "months",
        "y": "years",
    }

    arg_name = units_map.get(time_frame.time_frame_unit)
    if arg_name:
        delta = relativedelta(**{arg_name: time_frame.time_frame_interval})
        now = datetime.now()
        return (now + delta) - now

    return None
