#!/bin/bash
set -e

echo "Waiting for databases..."
until pg_isready -h "${GUARD_DATABASE_HOST}" -p 5432 -U "${GUARD_DATABASE_USER}"; do
  sleep 1
done

echo "Running migrations..."
cd guard
alembic upgrade head

echo "Starting guard server..."
exec uvicorn main:app --host 0.0.0.0 --port 8000

