#!/bin/bash
set -e

echo "Starting FastAPI server..."
exec uvicorn scavenger.api:app --host 0.0.0.0 --port 8000
