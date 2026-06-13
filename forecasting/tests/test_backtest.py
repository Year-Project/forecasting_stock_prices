import numpy as np
import pandas as pd
import pytest

from stock_forecast.backtest import run_panel_signal_backtest, run_per_ticker_signal_backtest


def test_per_ticker_signal_backtest_keeps_tickers_separate():
    predictions = pd.DataFrame(
        [
            {"date": "2024-01-01", "ticker": "A", "y_true": 0.02, "y_pred": 0.03},
            {"date": "2024-01-02", "ticker": "A", "y_true": -0.01, "y_pred": -0.02},
            {"date": "2024-01-01", "ticker": "B", "y_true": 0.01, "y_pred": -0.01},
            {"date": "2024-01-02", "ticker": "B", "y_true": 0.03, "y_pred": 0.04},
        ]
    )
    predictions["date"] = pd.to_datetime(predictions["date"])

    equity, metrics = run_per_ticker_signal_backtest(
        predictions,
        rebalance_every=1,
        transaction_cost_bps=0,
        slippage_bps=0,
    )

    assert set(metrics["ticker"]) == {"A", "B"}
    assert set(equity["ticker"]) == {"A", "B"}
    assert equity.loc[equity["ticker"] == "A", "position"].tolist() == [1, 0]
    assert equity.loc[equity["ticker"] == "B", "position"].tolist() == [0, 1]
    assert np.isfinite(metrics.loc[metrics["ticker"] == "A", "cumulative_return"]).all()


def test_panel_signal_backtest_builds_equal_weight_panel():
    predictions = pd.DataFrame(
        [
            {"date": "2024-01-01", "target_date": "2024-01-02", "ticker": "A", "model_name": "m", "y_true": 0.02, "y_pred": 0.03},
            {"date": "2024-01-01", "target_date": "2024-01-02", "ticker": "B", "model_name": "m", "y_true": 0.01, "y_pred": -0.01},
            {"date": "2024-01-02", "target_date": "2024-01-03", "ticker": "A", "model_name": "m", "y_true": -0.01, "y_pred": -0.02},
            {"date": "2024-01-02", "target_date": "2024-01-03", "ticker": "B", "model_name": "m", "y_true": 0.03, "y_pred": 0.04},
        ]
    )
    predictions["date"] = pd.to_datetime(predictions["date"])
    predictions["target_date"] = pd.to_datetime(predictions["target_date"])

    equity, metrics = run_panel_signal_backtest(
        predictions,
        horizon=1,
        transaction_cost_bps=0,
        slippage_bps=0,
    )

    assert set(equity["ticker"]) == {"__panel__"}
    assert metrics["ticker"].tolist() == ["__panel__"]
    assert equity["active_tickers"].tolist() == [2, 2]
    assert equity["net_return"].tolist() == pytest.approx([0.01, 0.015])


def test_per_ticker_signal_backtest_can_trade_with_negative_biased_predictions():
    predictions = pd.DataFrame(
        [
            {"date": "2024-01-01", "ticker": "A", "y_true": 0.02, "y_pred": -0.04},
            {"date": "2024-01-02", "ticker": "A", "y_true": -0.01, "y_pred": -0.03},
            {"date": "2024-01-03", "ticker": "A", "y_true": 0.03, "y_pred": -0.01},
        ]
    )
    predictions["date"] = pd.to_datetime(predictions["date"])

    equity, metrics = run_per_ticker_signal_backtest(
        predictions,
        rebalance_every=1,
        transaction_cost_bps=0,
        slippage_bps=0,
    )

    assert equity["signal_anchor"].tolist() == [0.0, -0.04, -0.035]
    assert equity["position"].tolist() == [0, 1, 1]
    assert metrics.loc[metrics["ticker"] == "A", "number_of_trades"].item() == 1


def test_per_ticker_signal_backtest_anchor_uses_only_past_predictions():
    base = pd.DataFrame(
        [
            {"date": "2024-01-01", "ticker": "A", "y_true": 0.01, "y_pred": 0.10},
            {"date": "2024-01-02", "ticker": "A", "y_true": 0.01, "y_pred": 0.20},
            {"date": "2024-01-03", "ticker": "A", "y_true": 0.01, "y_pred": 0.30},
        ]
    )
    changed_future = base.copy()
    changed_future.loc[2, "y_pred"] = -100.0
    base["date"] = pd.to_datetime(base["date"])
    changed_future["date"] = pd.to_datetime(changed_future["date"])

    base_equity, _ = run_per_ticker_signal_backtest(
        base,
        rebalance_every=1,
        transaction_cost_bps=0,
        slippage_bps=0,
    )
    changed_equity, _ = run_per_ticker_signal_backtest(
        changed_future,
        rebalance_every=1,
        transaction_cost_bps=0,
        slippage_bps=0,
    )

    cols = ["signal_anchor", "signal_threshold", "position"]
    assert base_equity.loc[:1, cols].equals(changed_equity.loc[:1, cols])
    assert base_equity["signal_anchor"].tolist()[:2] == [0.0, 0.10]


@pytest.mark.parametrize("signal_anchor", ["median", "mean"])
def test_old_scalar_signal_anchors_are_rejected(signal_anchor):
    predictions = pd.DataFrame(
        [
            {"date": "2024-01-01", "ticker": "A", "y_true": 0.01, "y_pred": -0.04},
            {"date": "2024-01-02", "ticker": "A", "y_true": 0.01, "y_pred": -0.03},
            {"date": "2024-01-03", "ticker": "A", "y_true": 0.01, "y_pred": -0.01},
        ]
    )
    predictions["date"] = pd.to_datetime(predictions["date"])

    with pytest.raises(ValueError, match="expanding_median"):
        run_per_ticker_signal_backtest(
            predictions,
            rebalance_every=1,
            transaction_cost_bps=0,
            slippage_bps=0,
            signal_anchor=signal_anchor,
        )


def test_expanding_mean_signal_anchor_uses_past_predictions():
    predictions = pd.DataFrame(
        [
            {"date": "2024-01-01", "ticker": "A", "y_true": 0.01, "y_pred": 0.10},
            {"date": "2024-01-02", "ticker": "A", "y_true": 0.01, "y_pred": 0.30},
            {"date": "2024-01-03", "ticker": "A", "y_true": 0.01, "y_pred": 0.50},
        ]
    )
    predictions["date"] = pd.to_datetime(predictions["date"])

    equity, _ = run_per_ticker_signal_backtest(
        predictions,
        rebalance_every=1,
        transaction_cost_bps=0,
        slippage_bps=0,
        signal_anchor="expanding_mean",
    )

    assert equity["signal_anchor"].tolist() == [0.0, 0.10, 0.20]


def test_per_ticker_signal_backtest_uses_target_date_as_realization_date():
    predictions = pd.DataFrame(
        [
            {
                "date": "2024-01-01",
                "entry_date": "2024-01-02",
                "target_date": "2024-01-04",
                "ticker": "A",
                "y_true": 0.10,
                "y_pred": 0.02,
            },
            {
                "date": "2024-01-02",
                "entry_date": "2024-01-03",
                "target_date": "2024-01-05",
                "ticker": "A",
                "y_true": -0.50,
                "y_pred": -0.01,
            },
        ]
    )
    for col in ["date", "entry_date", "target_date"]:
        predictions[col] = pd.to_datetime(predictions[col])

    equity, _ = run_per_ticker_signal_backtest(
        predictions,
        rebalance_every=1,
        transaction_cost_bps=0,
        slippage_bps=0,
        signal_anchor="zero",
    )

    assert equity["signal_date"].tolist() == [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")]
    assert equity["entry_date"].tolist() == [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]
    assert equity["target_date"].tolist() == [pd.Timestamp("2024-01-04"), pd.Timestamp("2024-01-05")]
    assert equity["date"].tolist() == equity["target_date"].tolist()
    assert equity["signal_anchor"].tolist() == [0.0, 0.0]
    assert equity["signal_threshold"].tolist() == [0.0, 0.0]
    assert equity["gross_return"].tolist() == [0.10, -0.0]
    assert np.isclose(equity["cumulative_return"].iloc[-1], np.exp(0.10) - 1.0)


def test_signal_backtest_annualizes_by_rebalance_trading_interval():
    predictions = pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-01", periods=10),
            "ticker": "A",
            "y_true": [0.01, 0.0, 0.0, 0.0, 0.0, 0.03, 0.0, 0.0, 0.0, 0.0],
            "y_pred": 1.0,
        }
    )

    equity, metrics = run_per_ticker_signal_backtest(
        predictions,
        rebalance_every=5,
        transaction_cost_bps=0,
        slippage_bps=0,
        signal_anchor="zero",
    )

    returns = equity["net_return"].to_numpy()
    periods_per_year = 252.0 / 5.0
    ticker_metrics = metrics.loc[metrics["ticker"] == "A"].iloc[0]
    assert ticker_metrics["periods_per_year"] == periods_per_year
    assert not np.isclose(ticker_metrics["periods_per_year"], 252.0 / 7.0)
    assert np.isclose(
        ticker_metrics["annualized_return"],
        np.exp(returns.mean() * periods_per_year) - 1.0,
    )
    assert np.isclose(
        ticker_metrics["annualized_volatility"],
        returns.std(ddof=1) * np.sqrt(periods_per_year),
    )


@pytest.mark.parametrize("kwargs", [{"rebalance_every": 0}, {"trading_days_per_year": 0}])
def test_signal_backtest_rejects_non_positive_annualization_inputs(kwargs):
    predictions = pd.DataFrame(
        [{"date": "2024-01-01", "ticker": "A", "y_true": 0.01, "y_pred": 0.02}]
    )
    predictions["date"] = pd.to_datetime(predictions["date"])

    with pytest.raises(ValueError, match="must be positive"):
        run_per_ticker_signal_backtest(predictions, **kwargs)
