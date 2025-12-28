from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
import uvicorn

from handlers import candles

load_dotenv()

app = FastAPI(title="Scavenger Service", description="API service to get tickers data")

app.include_router(candles.router, prefix="/scavenger")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
