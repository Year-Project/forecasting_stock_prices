# LSTM Research Plan For Return Forecasting

This note is scoped to the existing `forecasting/` pipeline. The current project predicts future log returns from daily OHLCV-derived features with strict chronological validation:

- Week horizon: `target_return_5_next_open`
- Month horizon: `target_return_21_next_open`
- Feature count: 127 numeric features per row
- Current tickers: `CBOM`, `MBNK`, `SBER`, `SBERP`, `SVCB`, `T`, `VTBR`
- Mature tickers have roughly 1,400 to 4,400 usable rows; `MBNK` and `SVCB` have only about 300 to 400 usable rows.

## Recommendation

The current implementation intentionally keeps one separate LSTM per ticker and horizon. Under that design, ticker identity is unnecessary: each model only sees one ticker, so a ticker embedding adds capacity without adding information.

Use each ticker as its own chronological sequence, build rolling windows of past features, tune hyperparameters with strict purged inner folds, then train a seed ensemble for the selected per-ticker configuration. For limited-history tickers, cap lookback, model size, batch size, and Optuna trials more aggressively than for mature tickers.

Recommended architecture:

```text
features[t-lookback+1:t]
-> optional input projection: Linear -> LayerNorm -> SiLU -> Dropout
-> 1 or 2 layer unidirectional LSTM
-> final hidden state
-> LayerNorm -> Dropout -> Linear return forecast
```

Use a stateless LSTM: reset hidden state for every window. That is easier to validate correctly and avoids hidden-state leakage between train, validation, and test periods.

Start with unidirectional LSTM. Bidirectional LSTM is not automatically leakage if the whole lookback window is known at prediction time, but it doubles capacity and often overfits small financial datasets. Treat it as a secondary search option only after the base model beats table baselines.

## Why This Model

Research supports using a standard gated LSTM rather than exotic variants. Greff et al. tested eight LSTM variants over 5,400 runs and found no variant significantly improved on the standard architecture, with the forget gate and output activation being the most critical components. Gers et al. introduced the forget gate for continual streams, which is relevant for financial sequences.

Finance-specific papers are more cautious: LSTMs can work, but they must be judged under walk-forward validation and against strong baselines. Fischer and Krauss applied LSTMs to S&P 500 constituents and framed the task as out-of-sample directional movement prediction. Baranochnikov and Slepaczuk used walk-forward validation to select LSTM/GRU architectures by validation-period investment performance. Recent sentiment studies show that OHLCV-only LSTM/GRU models are often unstable, and that news or other exogenous signals can materially improve results.

For this repository, the practical implication is:

- Predict future return, not raw close price.
- Use strict chronological and purged validation.
- Prefer small, regularized models.
- Use per-ticker LSTMs only with capped capacity and seed ensembling.
- Compare against naive, momentum, ridge, LightGBM, XGBoost, and CatBoost already present in the project.

## Training Protocol

1. Build windows inside each ticker only.
   - `X_i = feature rows from t-lookback+1 through t`
   - `y_i = target_return_horizon_next_open at t`
   - Skip windows crossing ticker boundaries.

2. Respect the current purge rule.
   - For validation tuning, train only on rows whose `target_date < validation_start`.
   - For final refit before test, train only on rows whose `target_date < test_start`.

3. Fit preprocessing only on training rows.
   - Impute feature medians from training rows.
   - Use `RobustScaler` or `StandardScaler` fit on training rows only.
   - Clip scaled features to about `[-5, 5]`.
   - Optionally train on standardized target returns, then invert predictions before metrics.

4. Loss and optimizer.
   - First choice: `SmoothL1Loss` or Huber loss because return targets have large outliers.
   - Secondary: `MSELoss`, useful only if validation RMSE is the main objective.
   - Optimizer: `AdamW`.
   - Clip gradients every step.
   - Use early stopping on validation loss plus validation directional metrics.

5. Validation objective.
   - Keep `directional_accuracy` as the primary model-selection metric.
   - Use purged inner walk-forward folds, preferably 3 to 5 folds.
   - Track RMSE and Spearman as diagnostics, not as multi-metric selection criteria.

6. Final prediction.
   - Refit using train plus validation data after purge.
   - Average predictions from seeds `[1, 7, 21, 42, 101]` for the selected hyperparameters.
   - Compare the ensemble against the prior single-seed LSTM and existing table baselines.

## Hyperparameter Search Space

Use Optuna with TPE. Start with 50 trials per horizon for a serious run; 10 to 15 trials are enough only for a smoke test. For limited-history tickers, cap model size and trials.

| Parameter | Recommended range | Notes |
| --- | --- | --- |
| `lookback` | categorical `[20, 40, 60, 90, 126]` | Use `[20, 40, 60]` for limited-history tickers. Avoid 252 unless validation later clearly supports it. |
| `input_projection_size` | categorical `[0, 32, 64, 128]` | `0` means feed all 127 features directly into the LSTM. Projection often regularizes. |
| `hidden_size` | categorical `[16, 32, 64, 96, 128]` | Remove 192 and avoid larger sizes unless per-ticker validation clearly supports them. |
| `num_layers` | categorical `[1, 2]` | One layer should be the default. Allow 2 layers only for mature tickers. |
| `lstm_dropout` | float `[0.0, 0.35]` | PyTorch applies this only between LSTM layers, so it matters only when `num_layers > 1`. |
| `head_dropout` | float `[0.15, 0.55]` | Important regularizer for noisy returns. |
| `learning_rate` | log float `[1e-4, 2e-3]` | AdamW default area is usually `3e-4` to `1e-3`. |
| `weight_decay` | log float `[1e-5, 1e-2]` | Often useful around `1e-4` to `1e-3`. |
| `batch_size` | categorical `[32, 64, 128]` | Use `[16, 32, 64]` for limited-history tickers. |
| `loss` | categorical `["smooth_l1", "mse"]` | Prefer `smooth_l1` first. |
| `huber_beta_week` | categorical `[0.02, 0.04, 0.06]` | Week target std is about `0.059`. |
| `huber_beta_month` | categorical `[0.04, 0.08, 0.12]` | Month target std is about `0.119`. |
| `feature_clip` | categorical `[3, 5, 8]` | Scaled feature clipping remains fitted only from train rows. |
| `grad_clip_norm` | float `[0.5, 2.0]` | Start with `1.0`. |
| `max_epochs` | fixed `100` to `200` | Early stopping should stop earlier. |
| `early_stop_patience` | integer `[8, 20]` | Use larger patience for month horizon. |
| `lr_scheduler_patience` | integer `[3, 8]` | `ReduceLROnPlateau` is enough. |

Narrow search for limited-history tickers:

```text
lookback: [20, 40, 60]
hidden_size: [16, 32, 64]
num_layers: [1]
head_dropout: 0.15 to 0.50
learning_rate: 1e-4 to 2e-3
weight_decay: 1e-5 to 1e-2
batch_size: [16, 32, 64]
```

## Experiments To Run

1. Per-ticker LSTM search-space refresh.
   - Train one model per ticker and horizon.
   - Use mature and limited-history search spaces.
   - Keep `directional_accuracy` as the primary validation metric.

2. Seed ensemble.
   - Take the best hyperparameter configuration.
   - Train seeds `[1, 7, 21, 42, 101]`.
   - Average return predictions.
   - This is usually a better use of compute than making the LSTM deeper.

3. Feature and target diagnostics.
   - Audit noisy or redundant LSTM features.
   - Try smaller feature subsets using only train/validation evidence.
   - Test volatility-normalized targets as a separate experiment.

4. Optional exogenous features.
   - If available, add market index, rates, FX, sector/industry, and news sentiment.
   - Literature suggests sentiment or other external signals can help more than changing LSTM variants.

## Acceptance Criteria

Do not judge the LSTM by validation loss alone. Accept it only if it improves the current strict reports on at least one of:

- Per-ticker test directional accuracy
- Test Spearman correlation
- Cost-adjusted test Sharpe or cumulative return
- Lower RMSE with no deterioration in directional accuracy

Also require:

- Leakage audit passes.
- Test predictions are generated only from final refit models.
- Results are compared against all current baselines.
- For limited-history tickers, LSTM must beat naive/momentum after costs; otherwise use fallback.

## Sources

- Hochreiter and Schmidhuber, "Long Short-Term Memory", Neural Computation, 1997: https://direct.mit.edu/neco/article/9/8/1735/6109/Long-Short-Term-Memory
- Gers, Schmidhuber, and Cummins, "Learning to Forget: Continual Prediction with LSTM", Neural Computation, 2000: https://pubmed.ncbi.nlm.nih.gov/11032042/
- Greff et al., "LSTM: A Search Space Odyssey", IEEE TNNLS, 2017: https://arxiv.org/abs/1503.04069
- Fischer and Krauss, "Deep learning with long short-term memory networks for financial market predictions", European Journal of Operational Research, 2018: https://www.sciencedirect.com/science/article/abs/pii/S0377221717310652
- Baranochnikov and Slepaczuk, "A Comparison of Long Short-Term Memory and Gated Recurrent Unit Models' Architectures with Novel Walk-Forward Approach to Algorithmic Investment Strategy", 2023: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4628576
- Dahal et al., "A comparative study on effect of news sentiment on stock price prediction with deep learning architecture", PLOS ONE, 2023: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0284695
- PyTorch `nn.LSTM` documentation: https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html
- PyTorch `AdamW` documentation: https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html
