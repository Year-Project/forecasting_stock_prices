from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker


class Database:
    def __init__(self, db_url: str):
        self.engine = create_async_engine(db_url, pool_pre_ping=True)
        self.sessionmaker = async_sessionmaker(self.engine, expire_on_commit=False)

    async def dispose(self):
        await self.engine.dispose()


class DatabaseRegistry:
    def __init__(self):
        self._databases: dict[str, Database] = {}

    def register(self, name: str, db_url: str):
        self._databases[name] = Database(db_url)

    def get(self, name: str) -> Database:
        return self._databases[name]


