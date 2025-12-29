from exceptions.exception_handler import BaseServiceException


class AdminAccessRequiredException(BaseServiceException):
    def __init__(self, message: str = "Admin access required", meta: dict | None = None, headers: dict | None = None):
        super().__init__(403, message, meta, headers)


class TokenExpiredException(BaseServiceException):
    def __init__(self, message: str = "Token is expired", meta: dict | None = None, headers: dict | None = None):
        super().__init__(401, message, meta, headers)


class InvalidTokenException(BaseServiceException):
    def __init__(self, message: str = "Invalid token", meta: dict | None = None):
        super().__init__(401, message, meta)
