from fastapi import Request
from fastapi.responses import JSONResponse


class BaseServiceException(Exception):
    def __init__(self, status_code: int, message: str = "", meta: dict | None = None,
                 headers: dict[str, str] | None = None):
        self.status_code = status_code
        self.message = message
        self.headers = headers

        if meta is None:
            meta = {}

        self.meta = meta

        super().__init__(message)

    def get_exception_details(self) -> dict:
        return {"message": self.message, "meta": self.meta}

    def get_exception_headers(self) -> dict[str, str] | None:
        return self.headers


async def service_exception_handler(request: Request, exception: BaseServiceException):
    return JSONResponse(
        status_code=exception.status_code,
        content=exception.get_exception_details(),
        headers=exception.get_exception_headers()
    )
