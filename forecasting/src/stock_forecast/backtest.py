from __future__ import annotations

import numpy as np
import pandas as pd


def run_per_ticker_signal_backtest(
    predictions: pd.DataFrame,
    rebalance_every: int = 5,
    transaction_cost_bps: float = 10.0,
    slippage_bps: float = 5.0,
    long_threshold: float = 0.0,
    signal_anchor: str = "expanding_median",
    date_col: str = "date",
    ticker_col: str = "ticker",
    target_col: str = "y_true",
    pred_col: str = "y_pred",
    trading_days_per_year: float = 252.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate each ticker independently with a long/cash forecast signal.

    Supported signal anchors are ``expanding_median`` (default),
    ``expanding_mean`` and ``zero``. Expanding anchors use only prior
    rebalance-date predictions for the same ticker.
    """
    if predictions.empty:
        return pd.DataFrame(), pd.DataFrame()
    if rebalance_every <= 0:
        raise ValueError("rebalance_every must be positive")
    if trading_days_per_year <= 0:
        raise ValueError("trading_days_per_year must be positive")

    data = predictions.sort_values([ticker_col, date_col]).copy()
    equity_frames = []
    metric_rows = []
    cost_rate = (transaction_cost_bps + slippage_bps) / 10000.0
    periods_per_year = float(trading_days_per_year) / float(rebalance_every)
    _validate_signal_anchor(signal_anchor)

    for ticker, group in data.groupby(ticker_col, sort=True):
        ticker_data = group.sort_values(date_col)
        dates = pd.DatetimeIndex(ticker_data[date_col].drop_duplicates().sort_values())
        rebalance_dates = dates[::rebalance_every]
        ticker_data = ticker_data[ticker_data[date_col].isin(rebalance_dates)].copy()
        ticker_data["_signal_anchor"] = _signal_anchors(ticker_data[pred_col], signal_anchor)

        rows = []
        previous_position = 0
        total_turnover = 0.0
        total_trades = 0
        for _, row in ticker_data.iterrows():
            if not np.isfinite(row[pred_col]) or not np.isfinite(row[target_col]):
                continue
            anchor = float(row["_signal_anchor"])
            signal_threshold = anchor + long_threshold
            position = int(float(row[pred_col]) > signal_threshold)
            turnover = abs(position - previous_position)
            gross_return = float(row[target_col]) * position
            net_return = gross_return - turnover * cost_rate
            signal_date = row[date_col]
            target_date = row.get("target_date", pd.NaT)
            realization_date = target_date if pd.notna(target_date) else signal_date
            equity_row = {
                "date": realization_date,
                "signal_date": signal_date,
                "ticker": ticker,
                "gross_return": gross_return,
                "net_return": net_return,
                "position": position,
                "turnover": float(turnover),
                "signal_anchor": anchor,
                "signal_threshold": signal_threshold,
            }
            if "entry_date" in ticker_data.columns:
                equity_row["entry_date"] = row["entry_date"]
            if "target_date" in ticker_data.columns:
                equity_row["target_date"] = target_date
            rows.append(equity_row)
            total_turnover += turnover
            total_trades += turnover
            previous_position = position

        equity = pd.DataFrame(rows)
        if equity.empty:
            metrics = _empty_metrics(periods_per_year)
        else:
            equity["cumulative_return"] = np.exp(equity["net_return"].cumsum()) - 1.0
            equity["wealth"] = 1.0 + equity["cumulative_return"]
            equity["drawdown"] = equity["wealth"] / equity["wealth"].cummax() - 1.0
            metrics = _equity_metrics(equity, int(total_trades), total_turnover, periods_per_year)
            equity_frames.append(equity)
        metrics["ticker"] = ticker
        metric_rows.append(metrics)

    equity_df = pd.concat(equity_frames, ignore_index=True) if equity_frames else pd.DataFrame()
    metrics_df = pd.DataFrame(metric_rows)
    return equity_df, metrics_df


def run_test_signal_backtest(
    validation_predictions: pd.DataFrame,
    test_predictions: pd.DataFrame,
    horizon: int,
    mode: str = "overlapping_tranches",
    transaction_cost_bps: float = 10.0,
    slippage_bps: float = 5.0,
    long_threshold: float = 0.0,
    signal_anchor: str = "expanding_median",
    date_col: str = "date",
    ticker_col: str = "ticker",
    model_col: str = "model_name",
    target_col: str = "y_true",
    pred_col: str = "y_pred",
    trading_days_per_year: float = 252.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate test predictions with thresholds seeded from validation predictions only."""
    if test_predictions.empty:
        return pd.DataFrame(), pd.DataFrame()
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if mode not in {"overlapping_tranches", "non_overlapping"}:
        raise ValueError("mode must be 'overlapping_tranches' or 'non_overlapping'")
    if trading_days_per_year <= 0:
        raise ValueError("trading_days_per_year must be positive")
    _validate_signal_anchor(signal_anchor)

    validation = validation_predictions.sort_values([ticker_col, model_col, date_col]).copy()
    test = test_predictions.sort_values([ticker_col, model_col, date_col]).copy()
    for frame in [validation, test]:
        if date_col in frame.columns:
            frame[date_col] = pd.to_datetime(frame[date_col])
        if "target_date" in frame.columns:
            frame["target_date"] = pd.to_datetime(frame["target_date"])
        if "entry_date" in frame.columns:
            frame["entry_date"] = pd.to_datetime(frame["entry_date"])

    cost_rate = (transaction_cost_bps + slippage_bps) / 10000.0
    weight = 1.0 / float(horizon) if mode == "overlapping_tranches" else 1.0
    periods_per_year = float(trading_days_per_year) if mode == "overlapping_tranches" else float(trading_days_per_year) / float(horizon)
    equity_frames = []
    metric_rows = []

    for (ticker, model_name), test_group in test.groupby([ticker_col, model_col], sort=True):
        group = test_group.sort_values(date_col).copy()
        if mode == "non_overlapping":
            rebalance_dates = pd.DatetimeIndex(group[date_col].drop_duplicates().sort_values())[::horizon]
            group = group[group[date_col].isin(rebalance_dates)].copy()
        validation_group = validation[
            (validation[ticker_col].astype(str) == str(ticker))
            & (validation[model_col].astype(str) == str(model_name))
        ].sort_values(date_col)
        seed_predictions = validation_group[pred_col] if pred_col in validation_group.columns else pd.Series(dtype=float)
        anchors = _seeded_signal_anchors(seed_predictions, group[pred_col], signal_anchor)

        rows = []
        previous_position = 0
        total_turnover = 0.0
        total_trades = 0
        for (_, row), anchor in zip(group.iterrows(), anchors):
            if not np.isfinite(row[pred_col]) or not np.isfinite(row[target_col]):
                continue
            threshold = float(anchor) + long_threshold
            position = int(float(row[pred_col]) > threshold)
            turnover = abs(position - previous_position)
            gross_return = float(row[target_col]) * position * weight
            net_return = gross_return - turnover * cost_rate * weight
            target_date = row.get("target_date", pd.NaT)
            realization_date = target_date if pd.notna(target_date) else row[date_col]
            equity_row = {
                "date": realization_date,
                "signal_date": row[date_col],
                "ticker": ticker,
                "model_name": model_name,
                "signal_mode": mode,
                "gross_return": gross_return,
                "net_return": net_return,
                "position": position,
                "capital_weight": weight,
                "turnover": float(turnover) * weight,
                "signal_anchor": float(anchor),
                "signal_threshold": threshold,
            }
            if "entry_date" in group.columns:
                equity_row["entry_date"] = row["entry_date"]
            if "target_date" in group.columns:
                equity_row["target_date"] = target_date
            rows.append(equity_row)
            total_turnover += turnover * weight
            total_trades += turnover
            previous_position = position

        equity = pd.DataFrame(rows)
        if equity.empty:
            metrics = _empty_metrics(periods_per_year)
        else:
            equity = (
                equity.groupby(["date", "ticker", "model_name", "signal_mode"], as_index=False)
                .agg(
                    gross_return=("gross_return", "sum"),
                    net_return=("net_return", "sum"),
                    turnover=("turnover", "sum"),
                    position=("position", "mean"),
                    capital_weight=("capital_weight", "sum"),
                    signal_date=("signal_date", "min"),
                    signal_anchor=("signal_anchor", "mean"),
                    signal_threshold=("signal_threshold", "mean"),
                )
                .sort_values("date")
            )
            equity["cumulative_return"] = np.exp(equity["net_return"].cumsum()) - 1.0
            equity["wealth"] = 1.0 + equity["cumulative_return"]
            equity["drawdown"] = equity["wealth"] / equity["wealth"].cummax() - 1.0
            metrics = _equity_metrics(equity, int(total_trades), total_turnover, periods_per_year)
            equity_frames.append(equity)
        metrics.update(
            {
                "ticker": ticker,
                "model_name": model_name,
                "signal_mode": mode,
                "n_rebalances": int(len(group)),
                "sample_warning": bool(mode == "non_overlapping" and len(group) < 12),
            }
        )
        metric_rows.append(metrics)

    equity_df = pd.concat(equity_frames, ignore_index=True) if equity_frames else pd.DataFrame()
    metrics_df = pd.DataFrame(metric_rows)
    return equity_df, metrics_df


def _equity_metrics(
    equity: pd.DataFrame,
    total_trades: int,
    total_turnover: float,
    periods_per_year: float,
) -> dict[str, float]:
    returns = equity["net_return"].astype(float)
    cumulative = float(equity["cumulative_return"].iloc[-1])
    ann_return = float(np.exp(returns.mean() * periods_per_year) - 1.0)
    ann_vol = float(returns.std(ddof=1) * np.sqrt(periods_per_year)) if len(returns) > 1 else float("nan")
    downside = returns[returns < 0]
    downside_vol = float(downside.std(ddof=1) * np.sqrt(periods_per_year)) if len(downside) > 1 else float("nan")
    sharpe = ann_return / ann_vol if ann_vol and np.isfinite(ann_vol) else float("nan")
    sortino = ann_return / downside_vol if downside_vol and np.isfinite(downside_vol) else float("nan")
    max_drawdown = float(equity["drawdown"].min())
    calmar = ann_return / abs(max_drawdown) if max_drawdown < 0 else float("nan")
    return {
        "cumulative_return": cumulative,
        "annualized_return": ann_return,
        "annualized_volatility": ann_vol,
        "periods_per_year": float(periods_per_year),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_drawdown": max_drawdown,
        "calmar": float(calmar),
        "turnover": float(total_turnover / len(equity)),
        "number_of_trades": int(total_trades),
    }


def _seeded_signal_anchors(
    seed_predictions: pd.Series,
    test_predictions: pd.Series,
    signal_anchor: str,
) -> list[float]:
    if signal_anchor == "zero":
        return [0.0] * len(test_predictions)

    history = [
        float(value)
        for value in pd.Series(seed_predictions).astype(float).tolist()
        if np.isfinite(value)
    ]
    anchors = []
    for value in pd.Series(test_predictions).astype(float).tolist():
        if not history:
            anchors.append(0.0)
        elif signal_anchor == "expanding_mean":
            anchors.append(float(np.mean(history)))
        elif signal_anchor == "expanding_median":
            anchors.append(float(np.median(history)))
        else:
            raise ValueError(
                "signal_anchor must be one of: 'zero', 'expanding_mean', 'expanding_median'"
            )
        if np.isfinite(value):
            history.append(float(value))
    return anchors


def _signal_anchors(predictions: pd.Series, signal_anchor: str) -> pd.Series:
    finite_predictions = pd.Series(predictions, index=predictions.index).astype(float)
    finite_predictions = finite_predictions.where(np.isfinite(finite_predictions))
    if signal_anchor == "zero":
        return pd.Series(0.0, index=predictions.index)

    past_predictions = finite_predictions.shift(1)
    if signal_anchor == "expanding_mean":
        anchors = past_predictions.expanding(min_periods=1).mean()
    elif signal_anchor == "expanding_median":
        anchors = past_predictions.expanding(min_periods=1).median()
    else:
        raise ValueError(
            "signal_anchor must be one of: 'zero', 'expanding_mean', 'expanding_median'"
        )
    return anchors.fillna(0.0)


def _validate_signal_anchor(signal_anchor: str) -> None:
    if signal_anchor not in {"zero", "expanding_mean", "expanding_median"}:
        raise ValueError(
            "signal_anchor must be one of: 'zero', 'expanding_mean', 'expanding_median'"
        )


def _empty_metrics(periods_per_year: float = float("nan")) -> dict[str, float]:
    keys = [
        "cumulative_return",
        "annualized_return",
        "annualized_volatility",
        "sharpe",
        "sortino",
        "max_drawdown",
        "calmar",
        "turnover",
        "number_of_trades",
    ]
    metrics = {key: float("nan") for key in keys}
    metrics["periods_per_year"] = float(periods_per_year)
    return metrics
