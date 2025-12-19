import redis.asyncio
from redis.asyncio import Redis

redis_pool = redis.asyncio.ConnectionPool(host='redis', port=6379, db=0, decode_responses=True)


async def get_redis_client() -> Redis:
    client = redis.asyncio.Redis(connection_pool=redis_pool)
    try:
        yield client
    finally:
        await client.close()
