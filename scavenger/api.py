from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
import uvicorn

from scavenger.handlers import candles
from utils.config_utils import ROOT_DIR

load_dotenv(Path(ROOT_DIR / ".env"))
load_dotenv()

app = FastAPI(title="Scavenger Service", description="API service to get tickers data")

app.include_router(candles.router, prefix="/scavenger")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
