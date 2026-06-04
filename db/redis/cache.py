import os

import redis.asyncio
from redis.asyncio import Redis


def create_redis_client(username: str, password: str) -> Redis:
    host = os.getenv("REDIS_HOST", "redis")
    port = int(os.getenv("REDIS_PORT", "6379"))
    pool = redis.asyncio.ConnectionPool(host=host, port=port, db=0, username=username, password=password,
                                        decode_responses=True)

    return redis.asyncio.Redis(connection_pool=pool)


class RedisClientProvider:
    def __init__(self, username: str, password: str):
        self.username = username
        self.password = password

    async def __call__(self) -> Redis:
        client = create_redis_client(self.username, self.password)
        try:
            yield client
        finally:
            await client.aclose()
