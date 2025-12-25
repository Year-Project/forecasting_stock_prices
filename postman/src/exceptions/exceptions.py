from exceptions.exception_handler import BaseServiceException


class InvalidTimeFrameException(BaseServiceException):
    def __init__(self, message: str = "", meta: dict | None = None):
        super().__init__(400, message, meta)
