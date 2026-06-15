# MLflow Integration Test Report

Date: 2026-06-03

Environment used:

- Conda env: `torchlab`
- Python: `3.12.12`
- MLflow: `3.13.0`
- pytest: `9.0.3`
- pandas: `2.3.3`
- numpy: `1.26.4`

## Dependency Setup

The `torchlab` environment initially had core ML packages, but was missing the service/test dependencies required for this integration:

- `pytest`
- `mlflow`
- `fastapi`
- `pydantic`
- `statsmodels`
- `aiokafka`
- `httpx`
- `python-dotenv`
- `sqlalchemy`
- `PyJWT`
- `redis`

These packages were installed into `torchlab` before running the tests.

## Commands Run

Static Python compile check:

```bash
source /home/sapce/miniconda3/etc/profile.d/conda.sh
conda activate torchlab
PYTHONPATH=forecasting/src:. python -m py_compile \
  forecasting/src/stock_forecast/mlflow_model.py \
  forecasting/src/stock_forecast/mlflow_cli.py \
  magician/services/forecast_service.py \
  magician/services/scavenger_client.py \
  magician/main.py \
  envoy/handlers/forecast.py \
  envoy/kafka/envoy_consumers.py \
  forecasting/tests/test_mlflow_model.py \
  magician/tests/test_forecast_service.py \
  postman/tests/test_forecast_return_schemas.py
```

Result: passed.

Docker Compose config validation:

```bash
docker compose config
docker compose --env-file .env_example config
```

Result: passed.

MLflow CLI availability:

```bash
PYTHONPATH=forecasting/src:. python -m stock_forecast.mlflow_cli --help
```

Result: passed. The CLI help is printed successfully.

Targeted integration tests:

```bash
PYTHONPATH=forecasting/src:. python -m pytest \
  forecasting/tests/test_mlflow_model.py \
  magician/tests/test_forecast_service.py \
  postman/tests/test_forecast_return_schemas.py \
  -q
```

Result: `5 passed, 1 warning`.

Full forecasting package tests:

```bash
PYTHONPATH=forecasting/src:. python -m pytest forecasting/tests -q
```

Result: `46 passed, 1 warning`.

Forecasting + service/schema tests:

```bash
PYTHONPATH=forecasting/src:. python -m pytest forecasting/tests magician/tests postman/tests -q
```

Result: `49 passed, 1 warning`.

Whitespace check:

```bash
git diff --check
```

Result: passed.

## Coverage

Verified:

- MLflow pyfunc router loads a bundled selected model and returns `forecast_return`.
- Unsupported ticker in pyfunc router fails with a clear error.
- `magician` uses MLflow route for daily weekly forecasts.
- `magician` computes `forecast_price = last_close * exp(forecast_return)`.
- `magician` falls back to `auto_arima` for non-daily requests.
- `forecast_return` survives Kafka response -> postman cache -> publish schema conversion.
- Existing forecasting tests still pass after adding MLflow modules.
- Root Docker Compose structure includes MLflow and validates with `.env_example`.

## Normal Application Flow Transcript

Because Docker daemon access was not available from the sandbox, the normal flow was tested in-process with the real `ForecastService` and the shared request/response schemas. External services were replaced with deterministic fakes:

- Scavenger candles returned close prices `[100.0, 105.0, 107.0]`.
- MLflow `stock_return_forecaster_week@prd` returned `forecast_return = 0.05`.
- `auto_arima` fallback returned `forecast_price = 123.45`.

### 1. Guard auth

Request:

```http
POST http://localhost:8001/guard/auth/v1/auth
Content-Type: application/json

{"telegram_id":742170129}
```

Response:

```json
{
  "status": 200,
  "body": {
    "access_token": "eyJ...test.jwt",
    "refresh_token": "omitted-for-report"
  }
}
```

### 2. Postman async forecast request

Request:

```http
POST http://localhost:8000/postman/forecasts/v1/forecast
Authorization: Bearer eyJ...test.jwt
Content-Type: application/json

{
  "isin": "SBER",
  "forecast_period": 7,
  "time_frame": "1d",
  "provide_plot": false
}
```

Response:

```json
{
  "status": 201,
  "body": null
}
```

Kafka request published by Postman:

```json
{
  "message_id": "00000000-0000-0000-0000-000000000001",
  "user_id": 742170129,
  "forecast_request": {
    "isin": "SBER",
    "forecast_period": 7,
    "time_frame": "1d",
    "provide_plot": false
  }
}
```

### 3. Magician MLflow route

Request:

```http
POST http://localhost:8002/magician/forecasts/v1/forecast
X-API-Key: <ADMIN_SECRET_KEY>
Content-Type: application/json

{
  "isin": "SBER",
  "forecast_period": 7,
  "time_frame": "1d",
  "provide_plot": false
}
```

Response:

```json
{
  "status": 200,
  "body": {
    "isin": "SBER",
    "forecast_period": 7,
    "time_frame": "1d",
    "forecast_price": 112.48600731223458,
    "forecast_return": 0.05,
    "forecast_confidence": null,
    "forecast_plot": null,
    "model": "models:/stock_return_forecaster_week@prd:ridge:SBER"
  }
}
```

Validated conversion:

```text
107.0 * exp(0.05) = 112.48600731223458
```

MLflow model URI loaded:

```json
["models:/stock_return_forecaster_week@prd"]
```

### 4. Kafka response and Postman cache

Kafka response consumed by Postman:

```json
{
  "message_id": "00000000-0000-0000-0000-000000000001",
  "user_id": 742170129,
  "forecast_response": {
    "isin": "SBER",
    "forecast_period": 7,
    "time_frame": "1d",
    "forecast_price": 112.48600731223458,
    "forecast_return": 0.05,
    "forecast_confidence": null,
    "forecast_plot": null,
    "model": "models:/stock_return_forecaster_week@prd:ridge:SBER"
  },
  "status": "completed"
}
```

Cached response stored by Postman:

```json
{
  "isin": "SBER",
  "forecast_period": 7,
  "time_frame": "1d",
  "forecast_price": 112.48600731223458,
  "forecast_return": 0.05,
  "forecast_confidence": null,
  "forecast_plot": null,
  "model": "models:/stock_return_forecaster_week@prd:ridge:SBER"
}
```

Envoy publish message:

```json
{
  "telegram_id": 742170129,
  "forecast_response": {
    "isin": "SBER",
    "forecast_period": 7,
    "time_frame": "1d",
    "forecast_price": 112.48600731223458,
    "forecast_return": 0.05,
    "forecast_confidence": null,
    "forecast_plot": null,
    "model": "models:/stock_return_forecaster_week@prd:ridge:SBER"
  }
}
```

### 5. Repeated Postman request after processing

Request:

```http
POST http://localhost:8000/postman/forecasts/v1/forecast
Authorization: Bearer eyJ...test.jwt
Content-Type: application/json

{
  "isin": "SBER",
  "forecast_period": 7,
  "time_frame": "1d",
  "provide_plot": false
}
```

Response:

```json
{
  "status": 200,
  "body": {
    "isin": "SBER",
    "forecast_period": 7,
    "time_frame": "1d",
    "forecast_price": 112.48600731223458,
    "forecast_return": 0.05,
    "forecast_confidence": null,
    "forecast_plot": null,
    "model": "models:/stock_return_forecaster_week@prd:ridge:SBER",
    "telegram_id": 742170129
  }
}
```

### 6. Fallback path for non-daily request

Request:

```http
POST http://localhost:8002/magician/forecasts/v1/forecast
X-API-Key: <ADMIN_SECRET_KEY>
Content-Type: application/json

{
  "isin": "SBER",
  "forecast_period": 7,
  "time_frame": "1w",
  "provide_plot": false
}
```

Response:

```json
{
  "status": 200,
  "body": {
    "isin": "SBER",
    "forecast_period": 7,
    "time_frame": "1w",
    "forecast_price": 123.45,
    "forecast_return": null,
    "forecast_confidence": null,
    "forecast_plot": null,
    "model": "auto_arima"
  }
}
```

Assertions passed:

- Daily `forecast_period = 7` uses `models:/stock_return_forecaster_week@prd`.
- MLflow return is exposed as `forecast_return`.
- `forecast_price` is computed from the last close and log return.
- `forecast_return` survives Kafka response, Postman cache, and Envoy publish schemas.
- Non-daily request falls back to `auto_arima`.

## Docker Compose Runtime Smoke

Runtime smoke was run through Docker after switching the command group with:

```bash
sg docker -c 'docker compose up --build -d'
```

All containers reached `Up` state:

- `guard_service`
- `postman_service`
- `magician_service`
- `scavenger_service`
- `mlflow`
- `redis`
- `kafka-1`, `kafka-2`, `kafka-3`
- `kafka-ui`
- `guard_db`, `postman_db`
- `envoy_service`

Runtime bugs found and fixed:

- Redis entrypoint had no executable bit and failed with `permission denied`.
- Redis ACL placeholders in `.env` generated invalid ACL syntax.
- `GUARD_DATABASE_HOST` and `POSTMAN_DATABASE_HOST` were missing, so entrypoints waited on `:5432`.
- Existing Postgres volumes had old passwords; roles were updated in-place with `ALTER USER`, without removing volumes.
- `envoy_service` restarted forever with placeholder `BOT_TOKEN`; default dev mode now sets `ENVOY_BOT_ENABLED=false`.
- Kafka topics were missing; services now create required topics at startup with `AIOKafkaAdminClient`.
- MLflow 3.13 rejected internal requests to `http://mlflow:5000`; compose now sets `--allowed-hosts` for `mlflow` and `localhost`.

Docker-network smoke requests:

```http
GET http://guard_service:8000/docs
GET http://postman_service:8000/docs
GET http://magician_service:8000/docs
GET http://scavenger_service:8000/docs
GET http://mlflow:5000
```

Result: all returned `200 OK`.

Auth request:

```http
POST http://guard_service:8000/guard/auth/v1/auth
Content-Type: application/json

{"telegram_id":742170129}
```

Result: `200 OK`, access and refresh tokens returned.

Direct candles request:

```http
GET http://scavenger_service:8000/scavenger/info/v1/candles?ticker=SBER&interval=24&start=2026-05-27&end=2026-06-03
```

Result: `200 OK`, 8 daily candles returned.

Postman forecast request:

```http
POST http://postman_service:8000/postman/forecasts/v1/forecast
Authorization: Bearer <access_token>
Content-Type: application/json

{
  "isin": "SBER",
  "forecast_period": 6,
  "time_frame": "1d",
  "provide_plot": false
}
```

Initial response:

```json
{
  "status": 201,
  "body": null
}
```

Cached response after async processing:

```json
{
  "status": 200,
  "body": {
    "isin": "SBER",
    "forecast_period": 6,
    "time_frame": "1d",
    "telegram_id": 742170129,
    "forecast_price": 323.5261304307609,
    "forecast_return": null,
    "forecast_confidence": null,
    "forecast_plot": null,
    "model": "auto_arima"
  }
}
```

MLflow UI is reachable from Docker network and the container is running. The MLflow model registry is currently empty, so `magician` logs:

```text
RESOURCE_DOES_NOT_EXIST: Registered Model with name=stock_return_forecaster_week not found
```

That fallback is expected until the training/register CLI is run and aliases are created:

```bash
docker compose run --rm magician_service \
  python -m stock_forecast.mlflow_cli \
  --tracking-uri http://mlflow:5000 \
  --experiment-name stock_return_forecasting \
  --alias prd \
  --horizons week month \
  --force-retrain
```

## Warnings

MLflow emits one warning during tests:

```text
Add type hints to the `predict` method to enable data validation and automatic signature inference during model logging.
```

This warning does not fail tests. The current pyfunc model intentionally keeps a broad `predict(context, model_input, params=None)` signature because it accepts pandas frames and dict/list-like input from MLflow runtime.

## Not Run

Full MLflow PRD model training and registration were not run in this pass because the repository does not contain ready strict-protocol artifacts under `forecasting/data/horizons/...`, and retraining can be long-running.

The executable smoke commands for a real environment are documented in:

- `docs/mlflow_service_deployment.md`

## Conclusion

The code-level, package-level, schema-level, Compose-configuration, Docker startup, docs, auth, candles, Kafka async forecast, and cache checks pass. The remaining validation step is running the MLflow training/register CLI so `models:/stock_return_forecaster_week@prd` and `models:/stock_return_forecaster_month@prd` exist and `magician` can use PRD models instead of `auto_arima` fallback.
