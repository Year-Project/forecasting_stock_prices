from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
import uvicorn
import os

from scavenger.src.handlers import candles
from utils.config_utils import ROOT_DIR

env_path = Path(ROOT_DIR / ".env")
if env_path.exists():
    load_dotenv(env_path)
else:
    load_dotenv()

app = FastAPI(
    title="Scavenger Service", 
    description="API service to get tickers data",
    version=os.getenv("API_VERSION", "0.0.1")
)

app.include_router(candles.router, prefix="/scavenger")

if __name__ == "__main__":
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8001"))
    
    uvicorn.run(
        app, 
        host=host, 
        port=port,
    )