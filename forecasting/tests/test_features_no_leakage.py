import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from stock_forecast.features import get_feature_columns, make_features
from stock_forecast.targets import make_future_return_target

NORMALIZED_FEATURES = {
    "atr_14_over_close",
    "macd_over_close",
    "macd_signal_over_close",
    "macd_hist_over_close",
    "rolling_std_5_over_20",
    "rolling_std_20_over_60",
    "rolling_std_5_over_60",
    "rolling_ret_mean_to_std_5",
    "rolling_ret_mean_to_std_10",
    "rolling_ret_mean_to_std_20",
    "rolling_ret_mean_to_std_60",
    "ret_1_z_20",
    "ret_1_z_60",
    "log_close_z_20",
    "log_close_z_60",
    "log_volume_z_20",
    "log_volume_z_60",
    "rolling_std_log_ratio_5_20",
    "rolling_std_log_ratio_20_60",
    "rolling_std_log_ratio_5_60",
    "close_over_sma_to_std_5",
    "close_over_sma_to_std_10",
    "close_over_sma_to_std_20",
    "close_over_sma_to_std_60",
    "rolling_ret_mean_diff_5_20",
    "rolling_ret_mean_diff_20_60",
    "rolling_ret_mean_diff_5_60",
    "rolling_ret_mean_5_to_std_20",
    "rolling_ret_mean_20_to_std_60",
    "rolling_ret_mean_5_to_std_60",
    "rolling_range_position_20",
    "rolling_range_position_60",
    "log_volume_z_abs_ret_z_20",
    "log_volume_z_abs_ret_z_60",
    "rsi_14_centered",
    "macd_hist_over_close_z_20",
    "macd_hist_over_close_z_60",
    "high_low_range_over_atr_14",
    "bollinger_position_20_centered",
}

ENGINEERED_FEATURES = {
    "realized_vol_5",
    "realized_vol_20",
    "realized_vol_60",
    "realized_vol_pct_rank_20_60",
    "realized_vol_5_over_20",
    "realized_vol_20_over_60",
    "drawdown_from_high_20",
    "drawdown_from_high_60",
    "distance_from_rolling_high_20",
    "distance_from_rolling_low_60",
    "rolling_max_drawdown_proxy_20",
    "rolling_max_drawdown_proxy_60",
    "dollar_volume",
    "dollar_volume_z_20",
    "dollar_volume_z_60",
    "volume_shock_abs_ret_20",
    "signed_volume_shock_20",
    "market_ret_equal_weight",
    "market_breadth_positive",
    "ret_minus_market",
    "cs_ret_rank",
    "cs_ret_pct_rank",
    "cs_volume_shock_rank_20",
    "cs_volume_shock_pct_rank_20",
    "market_beta_20",
    "market_beta_60",
    "residual_return",
    "residual_return_beta_20",
    "residual_return_beta_60",
    "residual_momentum_5",
    "residual_momentum_20",
    "residual_momentum_60",
}


def test_features_do_not_change_when_future_rows_change():
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    base = pd.DataFrame(
        {
            "date": dates,
            "ticker": "A",
            "open": range(100, 200),
            "high": range(101, 201),
            "low": range(99, 199),
            "close": range(100, 200),
            "volume": range(1000, 1100),
        }
    )
    changed = base.copy()
    changed.loc[70:, ["open", "high", "low", "close", "volume"]] *= 10

    base_features = make_features(base)
    changed_features = make_features(changed)
    feature_cols = get_feature_columns(base_features)

    assert_frame_equal(
        base_features.loc[:69, feature_cols],
        changed_features.loc[:69, feature_cols],
        check_dtype=False,
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )


def test_execution_metadata_is_not_used_as_features():
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    df = pd.DataFrame(
        {
            "date": dates,
            "ticker": "A",
            "open": range(100, 110),
            "high": range(101, 111),
            "low": range(99, 109),
            "close": range(100, 110),
            "volume": range(1000, 1010),
        }
    )
    featured = make_features(df, return_lags=[1], rolling_windows=[2], use_technical_indicators=False)
    targeted = make_future_return_target(featured, horizon=2, execution_timing="next_open")

    feature_cols = get_feature_columns(targeted)

    assert "entry_open" not in feature_cols
    assert "future_open" not in feature_cols
    assert "entry_date" not in feature_cols
    assert "target_return_2_next_open" not in feature_cols


def test_normalized_features_are_exposed_without_helper_columns():
    df = _feature_test_frame()
    featured = make_features(df)
    feature_cols = get_feature_columns(featured)

    assert NORMALIZED_FEATURES.issubset(feature_cols)
    assert ENGINEERED_FEATURES.issubset(feature_cols)
    assert "log_volume" not in featured.columns
    assert "log_close_sma_20" not in featured.columns
    assert "log_close_std_20" not in featured.columns
    assert "log_volume_sma_20" not in featured.columns
    assert "log_volume_std_20" not in featured.columns


def test_normalized_feature_values_match_expected_formulas():
    df = _feature_test_frame()
    featured = make_features(df)
    row = 70

    ret_1 = np.log(df["close"] / df["close"].shift(1))
    log_close = np.log(df["close"])
    log_volume = np.log(df["volume"])
    rolling_mean_20 = ret_1.rolling(window=20, min_periods=20).mean()
    rolling_mean_5 = ret_1.rolling(window=5, min_periods=5).mean()
    rolling_std_5 = ret_1.rolling(window=5, min_periods=5).std()
    rolling_std_20 = ret_1.rolling(window=20, min_periods=20).std()
    rolling_std_60 = ret_1.rolling(window=60, min_periods=60).std()
    close_over_sma_20 = df["close"] / df["close"].rolling(window=20, min_periods=20).mean() - 1.0
    rolling_min_60 = df["close"].rolling(window=60, min_periods=60).min()
    rolling_max_60 = df["close"].rolling(window=60, min_periods=60).max()
    close_window_60 = df["close"].iloc[row - 59 : row + 1].to_numpy()
    rolling_drawdowns_60 = close_window_60 / np.maximum.accumulate(close_window_60) - 1.0
    dollar_volume = df["close"] * df["volume"]
    dollar_volume_mean_20 = dollar_volume.rolling(window=20, min_periods=20).mean()
    dollar_volume_std_20 = dollar_volume.rolling(window=20, min_periods=20).std()
    log_close_mean_60 = log_close.rolling(window=60, min_periods=60).mean()
    log_close_std_60 = log_close.rolling(window=60, min_periods=60).std()
    log_volume_mean_20 = log_volume.rolling(window=20, min_periods=20).mean()
    log_volume_std_20 = log_volume.rolling(window=20, min_periods=20).std()
    macd_hist_over_close = featured["macd_hist_over_close"]
    macd_hist_over_close_mean_20 = macd_hist_over_close.rolling(window=20, min_periods=20).mean()
    macd_hist_over_close_std_20 = macd_hist_over_close.rolling(window=20, min_periods=20).std()

    assert np.isclose(featured.loc[row, "atr_14_over_close"], featured.loc[row, "atr_14"] / df.loc[row, "close"])
    assert np.isclose(featured.loc[row, "macd_over_close"], featured.loc[row, "macd"] / df.loc[row, "close"])
    assert np.isclose(featured.loc[row, "macd_signal_over_close"], featured.loc[row, "macd_signal"] / df.loc[row, "close"])
    assert np.isclose(featured.loc[row, "macd_hist_over_close"], featured.loc[row, "macd_hist"] / df.loc[row, "close"])
    assert np.isclose(featured.loc[row, "rolling_std_5_over_20"], rolling_std_5.loc[row] / rolling_std_20.loc[row])
    assert np.isclose(featured.loc[row, "rolling_std_20_over_60"], rolling_std_20.loc[row] / rolling_std_60.loc[row])
    assert np.isclose(featured.loc[row, "realized_vol_20"], rolling_std_20.loc[row] * np.sqrt(252.0))
    assert np.isclose(
        featured.loc[row, "realized_vol_20_over_60"],
        featured.loc[row, "realized_vol_20"] / featured.loc[row, "realized_vol_60"],
    )
    assert np.isclose(
        featured.loc[row, "rolling_ret_mean_to_std_20"],
        rolling_mean_20.loc[row] / rolling_std_20.loc[row],
    )
    assert np.isclose(featured.loc[row, "rolling_std_log_ratio_5_20"], np.log(rolling_std_5.loc[row] / rolling_std_20.loc[row]))
    assert np.isclose(
        featured.loc[row, "close_over_sma_to_std_20"],
        close_over_sma_20.loc[row] / rolling_std_20.loc[row],
    )
    assert np.isclose(
        featured.loc[row, "rolling_ret_mean_diff_5_20"],
        rolling_mean_5.loc[row] - rolling_mean_20.loc[row],
    )
    assert np.isclose(
        featured.loc[row, "rolling_ret_mean_20_to_std_60"],
        rolling_mean_20.loc[row] / rolling_std_60.loc[row],
    )
    assert np.isclose(
        featured.loc[row, "ret_1_z_20"],
        (ret_1.loc[row] - rolling_mean_20.loc[row]) / rolling_std_20.loc[row],
    )
    assert np.isclose(
        featured.loc[row, "log_close_z_60"],
        (log_close.loc[row] - log_close_mean_60.loc[row]) / log_close_std_60.loc[row],
    )
    assert np.isclose(
        featured.loc[row, "log_volume_z_20"],
        (log_volume.loc[row] - log_volume_mean_20.loc[row]) / log_volume_std_20.loc[row],
    )
    assert np.isclose(
        featured.loc[row, "rolling_range_position_60"],
        (df.loc[row, "close"] - rolling_min_60.loc[row]) / (rolling_max_60.loc[row] - rolling_min_60.loc[row]),
    )
    assert np.isclose(
        featured.loc[row, "drawdown_from_high_60"],
        df.loc[row, "close"] / rolling_max_60.loc[row] - 1.0,
    )
    assert np.isclose(
        featured.loc[row, "distance_from_rolling_low_60"],
        df.loc[row, "close"] / rolling_min_60.loc[row] - 1.0,
    )
    assert np.isclose(featured.loc[row, "rolling_max_drawdown_proxy_60"], rolling_drawdowns_60.min())
    assert np.isclose(featured.loc[row, "dollar_volume"], dollar_volume.loc[row])
    assert np.isclose(
        featured.loc[row, "dollar_volume_z_20"],
        (dollar_volume.loc[row] - dollar_volume_mean_20.loc[row]) / dollar_volume_std_20.loc[row],
    )
    assert np.isclose(
        featured.loc[row, "volume_shock_abs_ret_20"],
        featured.loc[row, "log_volume_z_20"] * abs(featured.loc[row, "ret_1"]),
    )
    assert np.isclose(
        featured.loc[row, "signed_volume_shock_20"],
        featured.loc[row, "log_volume_z_20"] * np.sign(featured.loc[row, "ret_1"]),
    )
    assert np.isclose(
        featured.loc[row, "log_volume_z_abs_ret_z_20"],
        featured.loc[row, "log_volume_z_20"] * abs(featured.loc[row, "ret_1_z_20"]),
    )
    assert np.isclose(
        featured.loc[row, "rsi_14_centered"],
        (featured.loc[row, "rsi_14"] - 50.0) / 50.0,
    )
    assert np.isclose(
        featured.loc[row, "macd_hist_over_close_z_20"],
        (
            macd_hist_over_close.loc[row]
            - macd_hist_over_close_mean_20.loc[row]
        )
        / macd_hist_over_close_std_20.loc[row],
    )
    assert np.isclose(
        featured.loc[row, "high_low_range_over_atr_14"],
        featured.loc[row, "high_low_range"] / featured.loc[row, "atr_14_over_close"],
    )
    assert np.isclose(
        featured.loc[row, "bollinger_position_20_centered"],
        featured.loc[row, "bollinger_position_20"] - 0.5,
    )
    assert np.isclose(featured.loc[row, "market_ret_equal_weight"], featured.loc[row, "ret_1"])
    assert np.isclose(featured.loc[row, "ret_minus_market"], 0.0)
    assert np.isclose(featured.loc[row, "cs_ret_pct_rank"], 1.0)


def test_cross_sectional_market_and_beta_features_use_current_universe_only():
    dates = pd.date_range("2024-01-01", periods=90, freq="D")
    rows = []
    for ticker, scale, volume_offset in [("A", 0.20, 0.0), ("B", 0.35, 40.0), ("C", -0.05, 80.0)]:
        idx = np.arange(len(dates), dtype=float)
        close = 100.0 + idx * scale + np.sin(idx / 5.0 + scale) * 0.8
        volume = 1000.0 + volume_offset + idx * (4.0 + scale)
        for row_idx, date in enumerate(dates):
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "open": close[row_idx] - 0.2,
                    "high": close[row_idx] + 0.5,
                    "low": close[row_idx] - 0.5,
                    "close": close[row_idx],
                    "volume": volume[row_idx],
                }
            )

    df = pd.DataFrame(rows)
    featured = make_features(df, return_lags=[1], rolling_windows=[5], use_technical_indicators=False)
    check_date = dates[70]
    date_rows = featured.loc[featured["date"].eq(check_date)].set_index("ticker")
    date_returns = date_rows["ret_1"]

    for ticker in ["A", "B", "C"]:
        expected_market = date_returns.drop(ticker).mean()
        assert np.isclose(date_rows.loc[ticker, "market_ret_equal_weight"], expected_market)
        assert np.isclose(date_rows.loc[ticker, "ret_minus_market"], date_returns.loc[ticker] - expected_market)

    assert np.isclose(date_rows["market_breadth_positive"].iloc[0], (date_returns > 0.0).mean())
    assert np.allclose(date_rows["cs_ret_pct_rank"], date_returns.rank(method="average", pct=True))
    assert np.allclose(
        date_rows["cs_volume_shock_pct_rank_20"],
        date_rows["log_volume_z_20"].rank(method="average", pct=True),
    )

    ticker_rows = featured.loc[featured["ticker"].eq("A")].sort_values("date").reset_index(drop=True)
    rolling_beta_20 = (
        ticker_rows["ret_1"].rolling(window=20, min_periods=20).cov(ticker_rows["market_ret_equal_weight"])
        / ticker_rows["market_ret_equal_weight"].rolling(window=20, min_periods=20).var()
    )
    residual_return = ticker_rows["ret_1"] - rolling_beta_20 * ticker_rows["market_ret_equal_weight"]
    assert np.isclose(ticker_rows.loc[70, "market_beta_20"], rolling_beta_20.loc[70])
    assert np.isclose(ticker_rows.loc[70, "residual_return"], residual_return.loc[70])
    assert np.isclose(
        ticker_rows.loc[70, "residual_momentum_5"],
        residual_return.rolling(window=5, min_periods=1).sum().loc[70],
    )


def test_normalized_features_use_nan_for_zero_denominators():
    dates = pd.date_range("2024-01-01", periods=80, freq="D")
    df = pd.DataFrame(
        {
            "date": dates,
            "ticker": "A",
            "open": 100.0,
            "high": 100.0,
            "low": 100.0,
            "close": 100.0,
            "volume": 1000.0,
        }
    )

    featured = make_features(df)

    assert pd.isna(featured.loc[70, "rolling_std_5_over_20"])
    assert pd.isna(featured.loc[70, "rolling_std_log_ratio_5_20"])
    assert pd.isna(featured.loc[70, "rolling_ret_mean_to_std_20"])
    assert pd.isna(featured.loc[70, "close_over_sma_to_std_20"])
    assert pd.isna(featured.loc[70, "ret_1_z_20"])
    assert pd.isna(featured.loc[70, "log_close_z_20"])
    assert pd.isna(featured.loc[70, "rolling_range_position_20"])
    assert pd.isna(featured.loc[70, "log_volume_z_20"])
    assert pd.isna(featured.loc[70, "macd_hist_over_close_z_20"])
    assert pd.isna(featured.loc[70, "high_low_range_over_atr_14"])


def _feature_test_frame() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=90, freq="D")
    idx = np.arange(len(dates), dtype=float)
    close = 100.0 + idx * 0.2 + np.sin(idx / 2.5) * 2.0
    return pd.DataFrame(
        {
            "date": dates,
            "ticker": "A",
            "open": close - 0.3,
            "high": close + 1.5,
            "low": close - 1.2,
            "close": close,
            "volume": 1000.0 + idx * 7.0 + np.cos(idx / 4.0) * 20.0,
        }
    )
