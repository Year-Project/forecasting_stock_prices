# Trading Metrics Audit

Sources:

- `forecasting/notebooks/03_model_comparison.ipynb`
- `forecasting/artifacts/reports/final_strict_comparison.md`
- `forecasting/artifacts/horizons/{week,month}/strict_protocol/reports/*`
- `envoy/handlers/forecast.py`
- `magician/services/forecast_service.py`

This is a research audit, not financial advice.

## Executive Assessment

The research pipeline is directionally well built for leakage control: it uses chronological splits, target-date purging, separate validation/test windows, explicit model artifacts, and transaction-cost-aware signal backtests. The major weakness is not data leakage; it is that the model-selection objective is not aligned with the trading objective. `directional_accuracy` is currently the primary validation metric, but in the generated report it has little to no relationship with realized out-of-sample Sharpe or cumulative return.

The current evidence supports treating the system as a forecasting research prototype, not yet as a production trading strategy. The strongest immediate improvement is to make model selection, threshold selection, and position sizing optimize a validation trading utility rather than raw prediction metrics.

## Current Experimental Design

- Universe: `CBOM`, `MBNK`, `SBER`, `SBERP`, `SVCB`, `T`, `VTBR`.
- Horizons: week, trained as 5 trading days; month, trained as 21 trading days.
- Target: next-open-to-future-open log return, with `entry_date`, `entry_open`, `future_open`, and `target_date` stored in prediction artifacts.
- Models: naive persistence, momentum, ridge, histogram gradient boosting, per-ticker LSTM, global LSTM all-features, global LSTM stationary-features.
- Selection metric: `directional_accuracy`.
- Headline signal mode: `overlapping_tranches`.
- Signal: long/cash, not long/short.
- Costs: 10 bps transaction cost plus 5 bps slippage.
- Signal anchor: expanding median of past predictions, seeded from validation for test.
- Test sample: roughly January 2026 to June 2026, with final target dates reaching June 13, 2026.

## What Is Strong

1. Leakage controls are materially better than a typical notebook backtest.

   The report's leakage audits pass for both week and month horizons. The protocol purges training rows whose target dates overlap the validation or test start.

2. The research distinguishes validation from test.

   Validation is used for model selection diagnostics; test is kept as the out-of-sample evaluation window.

3. Backtests include costs and turnover.

   This is important because the naive and momentum models often generate high turnover. The backtest already penalizes turnover with explicit costs.

4. The production router is artifact based.

   The MLflow pyfunc router loads selected strict-protocol final models by ticker, which is better than ad hoc notebook-to-service copying.

## Main Quantitative Findings

### Week Horizon

Best panel test results by model:

| Model | Cum. Return | Ann. Return | Sharpe | Max DD | Turnover | Trades |
|---|---:|---:|---:|---:|---:|---:|
| ridge | 7.70% | 15.21% | 3.27 | -2.45% | 3.32% | 149 |
| global_lstm_stationary | 1.30% | 2.49% | 2.38 | -0.17% | 0.18% | 8 |
| naive_persistence | -0.63% | -1.19% | -0.29 | -4.79% | 10.61% | 449 |
| momentum | -0.66% | -1.25% | -0.37 | -7.41% | 6.11% | 266 |
| hist_gradient_boosting | -2.55% | -4.81% | -1.10 | -5.41% | 1.36% | 62 |
| global_lstm_all | -2.22% | -4.19% | -1.75 | -4.77% | 3.45% | 137 |
| lstm | -3.31% | -6.22% | -2.60 | -6.03% | 2.86% | 128 |

Per-ticker validation-selected models were positive on 4 of 7 tickers in test. The strongest missed opportunity is that `ridge` was the best week panel model in test, but validation panel metrics were negative for all models, so the validation period did not identify this cleanly.

### Month Horizon

Best panel test results by model:

| Model | Cum. Return | Ann. Return | Sharpe | Max DD | Turnover | Trades |
|---|---:|---:|---:|---:|---:|---:|
| global_lstm_all | 4.37% | 8.51% | 8.37 | -0.30% | 0.18% | 33 |
| naive_persistence | 1.76% | 3.39% | 1.94 | -2.19% | 2.62% | 465 |
| ridge | 0.70% | 1.34% | 0.81 | -2.35% | 0.44% | 84 |
| momentum | 0.71% | 1.35% | 0.73 | -4.12% | 1.65% | 297 |
| lstm | -0.33% | -0.63% | -0.35 | -3.23% | 0.37% | 69 |
| hist_gradient_boosting | -0.55% | -1.05% | -0.57 | -4.56% | 0.17% | 32 |
| global_lstm_stationary | -0.45% | -0.85% | -1.97 | -0.98% | 0.19% | 34 |

Per-ticker validation-selected models were positive on only 2 of 7 tickers in test. The month result also shows an important model-selection failure: validation selected `global_lstm_stationary`, but test panel performance favored `global_lstm_all`. The high Sharpe for `global_lstm_all` is based on low volatility and only 33 trades, so it should be treated as promising but not statistically conclusive.

### Validation Does Not Predict Trading Metrics

Across ticker/model observations:

| Relationship | Week Corr. | Month Corr. |
|---|---:|---:|
| validation directional accuracy vs test Sharpe | -0.006 | 0.049 |
| validation directional accuracy vs test cumulative return | -0.042 | 0.049 |
| validation Sharpe vs test Sharpe | -0.043 | -0.543 |
| validation cumulative return vs test cumulative return | -0.257 | -0.314 |
| validation directional accuracy vs test directional accuracy | 0.326 | -0.369 |

This is the central audit finding: the current validation selection process is not stable enough to choose models for trading.

## Serving And Product Mismatch

`envoy/handlers/forecast.py` collects only:

- `isin`
- `forecast_period`
- `time_frame`
- `provide_plot`

That is enough to request a price forecast, but not enough to define a tradable decision. There is no user or strategy input for capital, risk budget, maximum drawdown, transaction cost model, minimum expected edge, benchmark, position sizing, or portfolio constraints.

The serving implementation also has a horizon/timeframe mismatch:

- The ML models are used only when `time_frame == "1d"`.
- Requests with `1m`, `10m`, `1h`, `1w`, or `1mo` fall back to AutoARIMA price forecasting.
- The ML week model is trained on a 5-trading-day horizon, but serving maps any daily `forecast_period <= 7` to the week model.
- The ML month model is trained on a 21-trading-day horizon, but serving maps any daily `forecast_period <= 31` to the month model.
- AutoARIMA returns only `forecast_price`; `forecast_return` is `None`, so it is not directly comparable with the trading backtests.

For trading metrics, the bot should not present unsupported timeframes as equivalent to the researched ML strategy.

## Main Risks

1. Objective mismatch.

   `directional_accuracy` treats a 1 bp correct call the same as a 500 bp correct call and ignores transaction costs, volatility, drawdown, and turnover. This explains why validation winners often do not become test PnL winners.

2. Single out-of-sample regime.

   The test window is about half a year. That is useful, but too small for a stable Sharpe estimate, especially for strategies with fewer than 50 trades.

3. Multiple testing.

   The workflow compares multiple models, horizons, tickers, LSTM variants, and thresholds. Without multiple-testing correction, the best-looking test result may be partly selection luck.

4. Limited-history names.

   `MBNK` and `SVCB` have limited history. Their selected models performed poorly in several test cases, which is consistent with unstable estimation.

5. Turnover and cost sensitivity.

   Naive persistence and momentum can trade often. Their economics will be very sensitive to spread, commission, slippage, liquidity, and execution assumptions.

6. Low-trade Sharpe inflation.

   The month `global_lstm_all` and `global_lstm_stationary` variants have very low turnover. High Sharpe with very low drawdown over a short sample can be mechanically inflated and should be validated over more regimes.

## Improvements To Prioritize

### 1. Replace Primary Selection Metric

Use validation trading utility as the primary selector:

```text
utility = Sharpe
          - lambda_drawdown * abs(max_drawdown)
          - lambda_turnover * turnover
          - lambda_instability * metric_std_across_folds
```

Apply hard constraints before ranking:

- minimum number of trades
- maximum drawdown limit
- maximum turnover limit
- positive net cumulative return after costs
- minimum validation coverage
- no selection if utility is not better than cash and buy-and-hold benchmarks

Keep `directional_accuracy`, `MAE`, `RMSE`, `Pearson`, and `Spearman` as diagnostics, not as the production selection target.

### 2. Add Multi-Fold Purged Walk-Forward Evaluation

One validation window and one test window are not enough. Add multiple anchored or rolling purged folds across earlier years. Report:

- mean and median Sharpe by fold
- worst-fold Sharpe
- hit rate of profitable folds
- fold-to-fold turnover stability
- deflated Sharpe or probabilistic Sharpe
- confidence intervals by block bootstrap

A model should be selected only if it is stable across folds, not just best in one recent validation period.

### 3. Tune The Trading Rule, Not Only The Model

For each model and ticker, tune signal conversion on validation:

- threshold grid based on predicted return quantiles
- no-trade band around zero or around expanding median
- hysteresis to reduce position flipping
- minimum expected net return after cost
- volatility-scaled threshold
- maximum holding overlap and capital usage

The signal should be:

```text
trade if expected_return - expected_cost - risk_buffer > 0
```

not simply:

```text
trade if prediction > expanding_median + fixed_threshold
```

### 4. Introduce Position Sizing

Current backtest is effectively binary long/cash. Improve it with:

- volatility targeting per ticker
- position size proportional to forecast edge divided by forecast volatility
- portfolio exposure cap
- ticker concentration cap
- liquidity cap based on average traded value
- drawdown-based de-risking

For example:

```text
raw_weight = forecast_return / forecast_volatility
weight = clip(raw_weight, 0, max_ticker_weight)
portfolio_weights = normalize_to_risk_budget(weight)
```

### 5. Evaluate Rank-Based Portfolios

For a small universe, raw return forecasts are noisy. Cross-sectional ranks may be more robust:

- long top 1 to 3 names by expected risk-adjusted return
- cash for names below threshold
- optional hedge with index or liquid benchmark
- pair or relative-value trades for related names such as `SBER` and `SBERP`

Report excess return versus equal-weight buy-and-hold and cash.

### 6. Add Benchmarks

Every report should include:

- cash
- buy-and-hold per ticker
- equal-weight universe
- equal-risk universe
- MOEX/IMOEX or relevant sector benchmark, if available
- simple trend following baseline

Trading metrics are only meaningful if the strategy clears these baselines after costs.

### 7. Improve Data Inputs

The current OHLCV feature set is a good base, but trading metrics will likely improve more from better data than from larger neural networks. Add:

- total-return adjustment for dividends
- corporate actions and split handling
- benchmark/index returns
- sector or industry returns
- rates, FX, oil, and macro proxies relevant to Russian equities
- bid/ask spread or realistic spread proxy
- suspended trading and limit-move flags
- free float, market cap, and liquidity filters
- broader ticker universe to reduce idiosyncratic overfit

### 8. Calibrate Forecasts

The strategy needs uncertainty, not only point forecasts. Add:

- probability of positive return
- expected shortfall or downside risk
- conformal intervals
- quantile regression
- calibration curves for predicted sign probability

The bot response should expose `forecast_return`, `probability_positive`, `expected_net_return`, `signal`, `position_size`, `model_name`, `horizon_name`, and `last_observation_date`.

### 9. Align Serving With Research Horizons

Restrict or relabel user choices:

- If ML trading signal is enabled, accept only daily data and exact supported horizons: 5 and 21 trading days.
- If the user requests unsupported `time_frame` or `forecast_period`, explicitly label the response as AutoARIMA price forecast, not a researched trading signal.
- Do not compare AutoARIMA outputs with ML strategy metrics unless AutoARIMA is included in the same strict backtest.

### 10. Add Live Paper Trading Monitoring

Start logging every forecast and its realized result:

- request timestamp
- ticker/ISIN
- model
- horizon
- prediction
- signal threshold
- signal
- realized return
- slippage estimate
- forecast error
- realized PnL

Then publish rolling live metrics: 20-trade Sharpe, cumulative PnL, drawdown, hit rate, average win/loss, turnover, and calibration drift.

## Recommended Next Research Iteration

1. Re-run strict protocol with a trading-utility primary metric instead of `directional_accuracy`.
2. Add at least 5 purged walk-forward folds and report fold stability.
3. Add benchmark and buy-and-hold comparisons to `03_model_comparison.ipynb`.
4. Backtest the exact MLflow router-selected models, not just individual model families.
5. Add threshold and no-trade-band selection under cost and turnover constraints.
6. Add volatility targeting and portfolio-level exposure caps.
7. Restrict the bot to researched daily horizons or clearly mark fallback forecasts as non-strategy outputs.

The most promising current candidates are week `ridge` at the panel level and month `global_lstm_all`, but neither should be promoted without multi-fold validation and benchmark-relative evidence.
