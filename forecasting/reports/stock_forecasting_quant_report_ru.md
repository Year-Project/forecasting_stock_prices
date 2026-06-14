# Прогнозирование доходности акций: количественный отчет

Дата отчета: 2026-06-14
PowerPoint: `stock_forecasting_quant_report_ru.pptx`

## Ключевые выводы

- Данные: 20,426 OHLCV-наблюдений, 7 тикеров, период 2007-05-28 - 2026-06-13.
- Горизонты: неделя = 5 торговых дней, месяц = 21 торговый день; target = future log return от next open.
- Anti-leakage audit: week 40/40, month 40/40.
- Лучший full-panel результат за неделю: Ridge, cumulative return +7.7%, Sharpe 3.27.
- Лучший full-panel результат за месяц: Global LSTM all, cumulative return +4.4%, Sharpe 8.37.
- Выбранные по directional accuracy модели положительны в тесте: week 4/7, month 2/7.

## Слайды

### 1. Титульный слайд

Проект: `forecasting/`. Горизонты: week = 5 торговых дней, month = 21 торговый день. Target: next-open-to-future-open log return.

### 2. Executive Summary

- Research-loop завершен: EDA, leakage-safe features, strict validation/test, signal backtest, графики и модельные артефакты.
- Anti-leakage checks: week 40/40, month 40/40.
- Главный вывод: production trading требует selector по trading utility, а не только directional accuracy.

### 3. Объем работ

Выполнены notebook stages: `01_eda`, LSTM/global LSTM/Mamba EDA, table/LSTM/global LSTM/Mamba forecasting, финальное сравнение моделей.

### 4. Данные и признаки

| Dataset | Rows / sequences | Features | Комментарий |
|---|---:|---:|---|
| Base table | 19,407 | 127 | rolling/cross-sectional OHLCV |
| Per-ticker LSTM | 19,407 | 160 | +33 sequence features |
| Global LSTM | 18,994 | 167 | ticker one-hot + pooled panel |
| Global Mamba | 18,532 | 275 | +108 Mamba-specific features |

### 5. Валидационный протокол

- Chronological train / validation / test; random split не используется.
- Purging по target dates защищает от leakage.
- Headline signal mode: `overlapping_tranches`.
- Costs: 10 bps transaction cost + 5 bps slippage.

### 6. Модельный периметр

Baselines: naive persistence, momentum. Tabular: ridge, HistGradientBoosting. Deep: per-ticker LSTM, global LSTM all/stationary, global Mamba all/stationary.

### 7. Week: full-panel results

| Model | Cum. return | Sharpe | Max DD | Turnover | Trades |
|---|---:|---:|---:|---:|---:|
| Ridge | +7.7% | 3.27 | -2.4% | 3.32% | 149 |
| Global LSTM stationary | +1.3% | 2.38 | -0.2% | 0.18% | 8 |
| Naive persistence | -0.6% | -0.29 | -4.8% | 10.61% | 449 |
| Momentum | -0.7% | -0.37 | -7.4% | 6.11% | 266 |
| Global LSTM all | -2.2% | -1.75 | -4.8% | 3.45% | 137 |
| HistGradientBoosting | -2.5% | -1.10 | -5.4% | 1.36% | 62 |
| LSTM per-ticker | -3.3% | -2.60 | -6.0% | 2.86% | 128 |

### 8. Month: full-panel results

| Model | Cum. return | Sharpe | Max DD | Turnover | Trades |
|---|---:|---:|---:|---:|---:|
| Global LSTM all | +4.4% | 8.37 | -0.3% | 0.18% | 33 |
| Naive persistence | +1.8% | 1.94 | -2.2% | 2.62% | 465 |
| Momentum | +0.7% | 0.73 | -4.1% | 1.65% | 297 |
| Ridge | +0.7% | 0.81 | -2.3% | 0.44% | 84 |
| LSTM per-ticker | -0.3% | -0.35 | -3.2% | 0.37% | 69 |
| Global LSTM stationary | -0.4% | -1.97 | -1.0% | 0.19% | 34 |
| HistGradientBoosting | -0.5% | -0.57 | -4.6% | 0.17% | 32 |

### 9. Selected models vs buy-and-hold

| Horizon | Positive selected models | Excess over buy-and-hold | Best selected | Worst selected |
|---|---:|---:|---|---|
| Week | 4/7 | 4/7 | VTBR +7.6% | SVCB -3.2% |
| Month | 2/7 | 4/7 | VTBR +10.8% | MBNK -3.5% |

### 10. Диагностика отбора

Directional accuracy слабо связан с торговой метрикой. В аудите корреляция validation directional accuracy vs test Sharpe равна -0.006 для week и 0.049 для month; validation Sharpe vs test Sharpe равна -0.043 и -0.543 соответственно.

### 11. Mamba extension

Week:

| Model | Positive tickers | Mean ticker return | Median Sharpe | Best ticker |
|---|---:|---:|---:|---|
| Global Mamba all | 4/7 | +5.2% | 2.10 | CBOM +33.9% |
| Global Mamba stationary | 2/7 | +4.0% | -0.79 | CBOM +39.0% |

Month:

| Model | Positive tickers | Mean ticker return | Median Sharpe | Best ticker |
|---|---:|---:|---:|---|
| Global Mamba all | 5/7 | +4.3% | 5.27 | CBOM +24.3% |
| Global Mamba stationary | 5/7 | +3.2% | 1.23 | CBOM +18.8% |

Примечание: mean return здесь является простым средним ticker-level metrics, не capital-weighted panel.

### 12. Production и риски

- Objective mismatch.
- Single out-of-sample regime.
- Multiple testing.
- Limited-history тикеры MBNK и SVCB.
- Serving/API mismatch между researched daily ML horizons и fallback AutoARIMA timeframes.

### 13. Рекомендации

1. Заменить основной selector на validation trading utility с штрафами за drawdown, turnover и нестабильность.
2. Добавить multi-fold purged walk-forward across regimes.
3. Оптимизировать signal rule и position sizing, а не только модель прогноза.
4. Синхронизировать serving API с реально исследованными горизонтами и daily timeframe.
