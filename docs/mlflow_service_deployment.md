# MLflow Service Deployment

This project starts all runtime services from the repository root with one Compose command. MLflow is part of the same Compose graph and is reachable by other services through the internal Docker network.
The Compose service uses MLflow artifact proxying, so training containers can upload run tables, plots and model artifacts through `http://mlflow:5000` without writing directly to the MLflow volume.

## Start

```bash
cp .env_example .env
```

Fill `.env` with Kafka, Redis, database and JWT values, then start everything:

```bash
docker compose up --build
docker compose ps
```

Useful web interfaces:

- MLflow: http://localhost:5000
- Kafka UI: http://localhost:8080
- Postman OpenAPI: http://localhost:8000/docs
- Guard OpenAPI: http://localhost:8001/docs
- Magician OpenAPI: http://localhost:8002/docs
- Scavenger OpenAPI: http://localhost:8003/docs

## Register MLflow Models

Run this after MLflow is up. From the `forecasting/` directory:

```bash
python3 -m pip install -e ".[boosting,deep]"
MLFLOW_TRACKING_URI=http://localhost:5000 \
stock-forecast-train-register-mlflow --artifact-root artifacts --skip-training
```

`--skip-training` registers the existing strict-protocol artifacts. To retrain before registration, remove `--skip-training`; add `--force-retrain` if cached hyperparameters/models must be ignored.

Expected registered model aliases:

- `models:/stock_return_forecaster_week@prd`
- `models:/stock_return_forecaster_month@prd`

After the command finishes, MLflow UI shows:

- parent runs named `week-strict-prd` and `month-strict-prd` in the `stock_return_forecasting` experiment;
- nested runs named `{horizon}-{ticker}-{model}` with per-model validation, test and signal metrics;
- run artifacts under `tables/`, `reports/`, `plots/`, `reproducibility/` and `model/`;
- registered pyfunc models in the Models tab with the `prd` alias.

## API Smoke Test

Get a user JWT from Guard:

```bash
ACCESS_TOKEN=$(
  curl -s -X POST http://localhost:8001/guard/auth/v1/auth \
    -H "Content-Type: application/json" \
    -d '{"telegram_id":742170129}' \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["access_token"])'
)
```

Request a weekly forecast through Postman:

```bash
curl -i -X POST http://localhost:8000/postman/forecasts/v1/forecast \
  -H "Authorization: Bearer ${ACCESS_TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{"isin":"SBER","forecast_period":7,"time_frame":"1d","provide_plot":false}'
```

Request a monthly forecast:

```bash
curl -i -X POST http://localhost:8000/postman/forecasts/v1/forecast \
  -H "Authorization: Bearer ${ACCESS_TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{"isin":"SBER","forecast_period":21,"time_frame":"1d","provide_plot":false}'
```

The first request usually returns `201 Created` because forecast processing is asynchronous through Kafka. Repeat the same request after processing; the cached response should return `200 OK` with `forecast_price`, `forecast_return`, and `model`.

Watch the async path:

```bash
docker compose logs -f magician_service postman_service
```

Check Scavenger candles directly:

```bash
curl "http://localhost:8003/scavenger/info/v1/candles?ticker=SBER&interval=24"
```

## MLflow Fallback Behavior

`magician` uses MLflow only for daily candles and supported horizons:

- `forecast_period <= 7`: `models:/stock_return_forecaster_week@prd`
- `forecast_period <= 31`: `models:/stock_return_forecaster_month@prd`

Unsupported timeframes, periods above 31, missing registry aliases, unsupported tickers, or MLflow prediction errors fall back to the existing `auto_arima` implementation. In fallback responses `forecast_return` is `null` and `model` is `auto_arima`.
