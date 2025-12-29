import redis.asyncio
from redis.asyncio import Redis

redis_pool = redis.asyncio.ConnectionPool(host='redis', port=6379, db=0, decode_responses=True)


class RedisClientProvider:
    def __init__(self, username: str, password: str):
        self.username = username
        self.password = password

    async def __call__(self) -> Redis:
        client = redis.asyncio.Redis(connection_pool=redis_pool, username=self.username, password=self.password)
        try:
            yield client
        finally:
            await client.aclose()
