from exceptions.exception_handler import BaseServiceException


class InvalidTimeFrameException(BaseServiceException):
    def __init__(self, message: str = "Invalid time frame provided", meta: dict | None = None,
                 headers: dict | None = None):
        super().__init__(400, message, meta, headers)
