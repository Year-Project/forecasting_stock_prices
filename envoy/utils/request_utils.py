import httpx
from envoy.auth.auth_client import AuthClient


class ForecastClient:
    def __init__(self, base_url: str, auth: AuthClient):
        self._base_url = base_url
        self._auth = auth

    async def _authorized_request(self, telegram_id: int, method: str, url: str, json: dict | None = None):
        token = await self._auth.get_access(telegram_id)
        response = None
        async with httpx.AsyncClient() as client:
            r = await client.request(method, url, headers={"Authorization": f"Bearer {token}"}, json=json)
            code = r.status_code

        if r.status_code in (401, 403):
            token = await self._auth.handle_unauthorized(telegram_id)

            async with httpx.AsyncClient() as client:
                r = await client.request(method, url, headers={"Authorization": f"Bearer {token}"}, json=json)
                code = r.status_code

        return code

    async def request_forecast(self, telegram_id: int, payload: dict):
        await self._authorized_request(telegram_id=telegram_id, method="POST",
                                       url=f"{self._base_url}/postman/forecasts/v1/forecast", json=payload)
