# Stock Profit Forecast

Notebook-based leakage-safe research pipeline for forecasting future stock log returns from OHLCV data and evaluating predictions per stock with walk-forward validation.

This is research code, not financial advice. The model forecasts future returns, not guaranteed profit.

## Notebooks

Run notebooks in order:

1. `notebooks/01_eda.ipynb`
   Loads and validates raw OHLCV data, builds leakage-safe features and future log-return target, then saves prepared data artifacts.
2. `notebooks/01b_lstm_eda.ipynb`
   Adds leakage-safe LSTM sequence features and sequence diagnostics for week and month horizons.
3. `notebooks/02_table_model_forecasting.ipynb`
   Loads EDA artifacts, tunes table models with Optuna, trains week and month horizon models, caches best hyperparameters and saves out-of-fold predictions, metrics and fold models.
4. `notebooks/02b_lstm_forecasting.ipynb`
   Trains per-ticker/per-horizon LSTMs with strict purged validation, capped limited-history search spaces, and post-selection seed ensembles.
5. `notebooks/03_model_comparison.ipynb`
   Loads trained horizon artifacts, compares models per stock, runs independent per-stock signal backtests and saves final comparison reports and plots for each horizon.

## Data

The notebooks currently point to the existing repository data:

```text
../ML/gen/historical_data_1d.csv
```

Expected raw OHLCV fields are mapped in `01_eda.ipynb`:

```text
begin,name,open,high,low,close,volume
```

The internal clean schema is:

```text
date,ticker,open,high,low,close,volume
```

## Artifacts

The notebook chain writes:

- `artifacts/data/clean_ohlcv.parquet`
- `artifacts/data/model_dataset.parquet`
- `artifacts/data/feature_columns.json`
- `artifacts/data/horizons/{week,month}/model_dataset.parquet`
- `artifacts/data/horizons/{week,month}/feature_columns.json`
- `artifacts/data/lstm/horizons/{week,month}/model_dataset.parquet`
- `artifacts/data/lstm/horizons/{week,month}/sequence_diagnostics.json`
- `artifacts/horizons/{week,month}/hyperparams/{model}/*.json`
- `artifacts/horizons/{week,month}/models/{model}/.../*.pkl`
- `artifacts/horizons/{week,month}/lstm_only/strict_protocol/reports/*`
- `artifacts/horizons/{week,month}/predictions/*_oof.parquet`
- `artifacts/horizons/{week,month}/reports/*`
- `artifacts/horizons/{week,month}/plots/*.png`
- `artifacts/horizons/{week,month}/plots/tickers/*.png`

## Run Checks

From the `forecasting/` directory:

```bash
PYTHONPATH=src pytest
```

Random split is intentionally not used. Validation is walk-forward only, and per-stock signal metrics use out-of-fold predictions.
Signal backtests use `signal_anchor="expanding_median"` by default, so each threshold is calibrated only from past rebalance predictions for the same stock. Supported anchors are `expanding_median`, `expanding_mean` and `zero`.

## MLflow

After strict-protocol artifacts exist, register them and publish training results to MLflow:

```bash
MLFLOW_TRACKING_URI=http://localhost:5000 \
stock-forecast-train-register-mlflow --artifact-root artifacts --skip-training
```

The command logs parent runs, per-ticker/model nested runs, metrics tables, report artifacts, model artifacts, and registered pyfunc models for the week and month horizons.
