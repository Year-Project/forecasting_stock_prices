import json
import logging
import datetime as dt
from logging import Formatter

from typing import override


class JsonFormatter(Formatter):
    def __init__(self, *, fmt_keys: dict[str, str] = None):
        super().__init__()
        self.fmt_keys = fmt_keys if fmt_keys is not None else {}

    @override
    def format(self, record: logging.LogRecord) -> str:
        obligatory_fields = {
            "message": record.getMessage(),
            "timestamp": dt.datetime.fromtimestamp(
                record.created, tz=dt.timezone.utc
            ).isoformat(),
        }

        if record.exc_info is not None:
            obligatory_fields["exc_info"] = self.formatException(record.exc_info)

        if record.stack_info is not None:
            obligatory_fields["stack_info"] = self.formatStack(record.stack_info)

        message = {
            key: msg_val
            if (msg_val := obligatory_fields.pop(val, None)) is not None
            else getattr(record, val)
            for key, val in self.fmt_keys.items()
        }

        message.update(obligatory_fields)

        return json.dumps(message, default=str)
