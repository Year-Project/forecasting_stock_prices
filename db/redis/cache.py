import redis.asyncio
from redis.asyncio import Redis

redis_pool = redis.asyncio.ConnectionPool(host='redis', port=6379, db=0, decode_responses=True)


def create_redis_client(username: str, password: str) -> Redis:
    return redis.asyncio.Redis(connection_pool=redis_pool, username=username, password=password)


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
