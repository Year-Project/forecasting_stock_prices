#!/bin/bash
set -e

echo "Waiting for databases..."
until pg_isready -h "${POSTMAN_DATABASE_HOST}" -p 5432 -U "${POSTMAN_DATABASE_USER}"; do
  sleep 1
done

echo "Running migrations..."
cd postman
alembic upgrade head

echo "Starting postman server..."
exec uvicorn main:app --host 0.0.0.0 --port 8000