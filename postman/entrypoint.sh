#!/bin/bash
set -e

echo "Waiting for databases..."
until pg_isready -h "${DATABASE_HOST}" -p "${DATABASE_PORT}" -U "${DATABASE_USER}"; do
  sleep 1
done

echo "Running migrations..."
alembic upgrade head

echo "Starting FastAPI server..."
exec uvicorn postman.main:app --host 0.0.0.0 --port ${API_PORT}