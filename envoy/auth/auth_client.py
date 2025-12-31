import httpx
from dataclasses import dataclass


@dataclass
class TokenPair:
    access: str
    refresh: str | None


class AuthClient:
    def __init__(self, base_url: str):
        self._base_url = base_url
        self._tokens: dict[int, TokenPair] = {}

    async def _auth(self, telegram_id: int):
        async with httpx.AsyncClient() as client:
            r = await client.post(f"{self._base_url}/guard/auth/v1/auth", json={"telegram_id": telegram_id})

            r.raise_for_status()

        data = r.json()

        self._tokens[telegram_id] = TokenPair(access=data["access_token"], refresh=data.get("refresh_token"),)

    async def _refresh(self, refresh_token: str) -> TokenPair | None:
        async with httpx.AsyncClient() as client:
            r = await client.post(f"{self._base_url}/guard/auth/v1/refresh", json={"refresh_token": refresh_token})

        if r.status_code != 200:
            return None

        data = r.json()

        return TokenPair(access=data["access_token"], refresh=data.get("refresh_token"))

    async def get_access(self, telegram_id: int) -> str:
        pair = self._tokens.get(telegram_id)

        if pair is not None:
            return pair.access

        await self._auth(telegram_id)

        return self._tokens[telegram_id].access

    async def handle_unauthorized(self, telegram_id: int) -> str:
        pair = self._tokens.get(telegram_id)

        if pair is not None and pair.refresh is not None:
            refreshed = await self._refresh(pair.refresh)

            if refreshed is not None:
                self._tokens[telegram_id] = refreshed

                return refreshed.access

        await self._auth(telegram_id)

        return self._tokens[telegram_id].access
