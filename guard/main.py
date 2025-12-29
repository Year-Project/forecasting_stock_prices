import logging.config
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
import uvicorn

from db.redis.cache import redis_pool
from dependencies.dependencies import db_registry
from guard.src.handlers.v1 import auth_handler
from guard.src.handlers.v1 import internal_admin_handler
from exceptions.exception_handler import service_exception_handler, BaseServiceException
from utils.config_utils import load_yaml_config, ROOT_DIR

load_dotenv(Path(ROOT_DIR / ".env"))
load_dotenv()

LOGGER_CONFIG_PATH = Path(ROOT_DIR / "logger/config")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # app startup
    logging.config.dictConfig(
        load_yaml_config(
            f"{LOGGER_CONFIG_PATH}/logger_conf.{os.getenv('ENV', 'testing')}.yaml"
        )
    )
    db_registry.register("auth", os.getenv("DATABASE_URL"))

    try:
        yield
    finally:
        # app shutdown
        await redis_pool.disconnect()


app = FastAPI(title="Guard Service", description="Authentication and authorization service", lifespan=lifespan)

app.include_router(auth_handler.router, prefix="/guard")
app.include_router(internal_admin_handler.router, prefix="/guard")
app.add_exception_handler(BaseServiceException, service_exception_handler)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)

