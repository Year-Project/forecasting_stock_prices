from exceptions.exception_handler import BaseServiceException


class InvalidCredentialsException(BaseServiceException):
    def __init__(self, message: str = "Invalid credentials", meta: dict | None = None, headers: dict | None = None):
        super().__init__(401, message, meta, headers)


class UserNotFound(BaseServiceException):
    def __init__(self, message: str = "User not found", meta: dict | None = None, headers: dict | None = None):
        super().__init__(404, message, meta, headers)


class TokenRevokedException(BaseServiceException):
    def __init__(self, message: str = "Token was revoked", meta: dict | None = None, headers: dict | None = None):
        super().__init__(401, message, meta, headers)


