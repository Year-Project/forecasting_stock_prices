from __future__ import annotations

import numpy as np
import pandas as pd

TARGET_EXCLUDE_PREFIXES = ("target_", "future_", "entry_")
RAW_EXCLUDE_COLUMNS = {"date", "ticker", "open", "high", "low", "close", "volume"}
TRADING_DAYS_PER_YEAR = 252.0
VOLATILITY_WINDOWS = (5, 20, 60)
VOLATILITY_RANK_WINDOW = 60
DRAWDOWN_WINDOWS = (20, 60)
LIQUIDITY_WINDOWS = (20, 60)
BETA_WINDOWS = (20, 60)
RESIDUAL_MOMENTUM_WINDOWS = (5, 20, 60)


def make_features(
    df: pd.DataFrame,
    return_lags: list[int] | None = None,
    rolling_windows: list[int] | None = None,
    use_technical_indicators: bool = True,
) -> pd.DataFrame:
    lags = return_lags or [1, 2, 3, 5, 10, 20]
    windows = rolling_windows or [5, 10, 20, 60]

    out = df.sort_values(["ticker", "date"]).copy()
    pieces = []
    for _, group in out.groupby("ticker", sort=False, group_keys=False):
        pieces.append(_make_group_features(group.copy(), lags, windows, use_technical_indicators))
    out = pd.concat(pieces, ignore_index=True)
    out = _add_cross_sectional_features(out)
    out = _add_market_residual_features(out)
    return out.replace([np.inf, -np.inf], np.nan)


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = RAW_EXCLUDE_COLUMNS
    return [
        col
        for col in df.columns
        if col not in excluded
        and not col.startswith(TARGET_EXCLUDE_PREFIXES)
        and pd.api.types.is_numeric_dtype(df[col])
    ]


def _make_group_features(
    group: pd.DataFrame,
    lags: list[int],
    windows: list[int],
    use_technical_indicators: bool,
) -> pd.DataFrame:
    close = group["close"]
    high = group["high"]
    low = group["low"]
    volume = group["volume"]

    group["log_close"] = np.log(close)
    group["ret_1"] = np.log(close / close.shift(1))
    group["open_close_ret"] = np.log(close / group["open"])
    group["high_low_range"] = (high - low) / close
    group["close_to_high"] = close / high - 1.0
    group["close_to_low"] = close / low - 1.0

    return_rolls: dict[int, tuple[pd.Series, pd.Series]] = {}
    close_over_sma: dict[int, pd.Series] = {}
    for lag in lags:
        group[f"ret_lag_{lag}"] = group["ret_1"].shift(lag)

    for window in windows:
        roll_ret = group["ret_1"].rolling(window=window, min_periods=window)
        rolling_mean = roll_ret.mean()
        rolling_std = roll_ret.std()
        return_rolls[window] = (rolling_mean, rolling_std)
        group[f"rolling_ret_mean_{window}"] = rolling_mean
        group[f"rolling_ret_std_{window}"] = rolling_std
        sma = close.rolling(window=window, min_periods=window).mean()
        close_over_sma[window] = close / sma - 1.0
        group[f"close_over_sma_{window}"] = close_over_sma[window]

    for window in {5, 10, 20, 60}:
        if window not in return_rolls:
            roll_ret = group["ret_1"].rolling(window=window, min_periods=window)
            return_rolls[window] = (roll_ret.mean(), roll_ret.std())
        if window not in close_over_sma:
            sma = close.rolling(window=window, min_periods=window).mean()
            close_over_sma[window] = close / sma - 1.0

    group["rolling_std_5_over_20"] = _safe_divide(return_rolls[5][1], return_rolls[20][1])
    group["rolling_std_20_over_60"] = _safe_divide(return_rolls[20][1], return_rolls[60][1])
    group["rolling_std_5_over_60"] = _safe_divide(return_rolls[5][1], return_rolls[60][1])
    group["rolling_std_log_ratio_5_20"] = np.log(group["rolling_std_5_over_20"])
    group["rolling_std_log_ratio_20_60"] = np.log(group["rolling_std_20_over_60"])
    group["rolling_std_log_ratio_5_60"] = np.log(group["rolling_std_5_over_60"])
    for window in [5, 10, 20, 60]:
        group[f"rolling_ret_mean_to_std_{window}"] = _safe_divide(
            return_rolls[window][0],
            return_rolls[window][1],
        )
        group[f"close_over_sma_to_std_{window}"] = _safe_divide(
            close_over_sma[window],
            return_rolls[window][1],
        )
    group["rolling_ret_mean_diff_5_20"] = return_rolls[5][0] - return_rolls[20][0]
    group["rolling_ret_mean_diff_20_60"] = return_rolls[20][0] - return_rolls[60][0]
    group["rolling_ret_mean_diff_5_60"] = return_rolls[5][0] - return_rolls[60][0]
    group["rolling_ret_mean_5_to_std_20"] = _safe_divide(return_rolls[5][0], return_rolls[20][1])
    group["rolling_ret_mean_20_to_std_60"] = _safe_divide(return_rolls[20][0], return_rolls[60][1])
    group["rolling_ret_mean_5_to_std_60"] = _safe_divide(return_rolls[5][0], return_rolls[60][1])
    group["ret_1_z_20"] = _safe_divide(group["ret_1"] - return_rolls[20][0], return_rolls[20][1])
    group["ret_1_z_60"] = _safe_divide(group["ret_1"] - return_rolls[60][0], return_rolls[60][1])

    realized_vols: dict[int, pd.Series] = {}
    for window in VOLATILITY_WINDOWS:
        realized_vol = return_rolls[window][1] * np.sqrt(TRADING_DAYS_PER_YEAR)
        realized_vols[window] = realized_vol
        group[f"realized_vol_{window}"] = realized_vol
        group[f"realized_vol_pct_rank_{window}_{VOLATILITY_RANK_WINDOW}"] = _rolling_percentile_rank(
            realized_vol,
            window=VOLATILITY_RANK_WINDOW,
        )
    group["realized_vol_5_over_20"] = _safe_divide(realized_vols[5], realized_vols[20])
    group["realized_vol_20_over_60"] = _safe_divide(realized_vols[20], realized_vols[60])
    group["realized_vol_5_over_60"] = _safe_divide(realized_vols[5], realized_vols[60])

    log_close = group["log_close"]
    for window in [20, 60]:
        rolling_log_close = log_close.rolling(window=window, min_periods=window)
        group[f"log_close_z_{window}"] = _safe_divide(
            log_close - rolling_log_close.mean(),
            rolling_log_close.std(),
        )

    for window in DRAWDOWN_WINDOWS:
        rolling_max = close.rolling(window=window, min_periods=window).max()
        rolling_min = close.rolling(window=window, min_periods=window).min()
        group[f"drawdown_from_high_{window}"] = close / rolling_max - 1.0
        group[f"distance_from_rolling_high_{window}"] = group[f"drawdown_from_high_{window}"]
        group[f"distance_from_rolling_low_{window}"] = close / rolling_min - 1.0
        group[f"close_over_rolling_max_{window}"] = group[f"distance_from_rolling_high_{window}"]
        group[f"close_over_rolling_min_{window}"] = group[f"distance_from_rolling_low_{window}"]
        group[f"rolling_range_position_{window}"] = _safe_divide(
            close - rolling_min,
            rolling_max - rolling_min,
        )
        group[f"rolling_max_drawdown_proxy_{window}"] = _rolling_max_drawdown(close, window=window)

    nonzero_volume = volume.replace(0, np.nan)
    group["volume_change_1"] = np.log(nonzero_volume / nonzero_volume.shift(1))
    group["volume_over_sma_20"] = volume / volume.rolling(window=20, min_periods=20).mean() - 1.0
    group["volume_over_sma_60"] = volume / volume.rolling(window=60, min_periods=60).mean() - 1.0
    log_volume = np.log(nonzero_volume)
    group["dollar_volume"] = close * volume
    for window in LIQUIDITY_WINDOWS:
        rolling_log_volume = log_volume.rolling(window=window, min_periods=window)
        group[f"log_volume_z_{window}"] = _safe_divide(
            log_volume - rolling_log_volume.mean(),
            rolling_log_volume.std(),
        )
        rolling_dollar_volume = group["dollar_volume"].rolling(window=window, min_periods=window)
        group[f"dollar_volume_z_{window}"] = _safe_divide(
            group["dollar_volume"] - rolling_dollar_volume.mean(),
            rolling_dollar_volume.std(),
        )
        group[f"volume_shock_abs_ret_{window}"] = group[f"log_volume_z_{window}"] * group["ret_1"].abs()
        group[f"signed_volume_shock_{window}"] = group[f"log_volume_z_{window}"] * np.sign(group["ret_1"])
        group[f"dollar_volume_z_abs_ret_{window}"] = group[f"dollar_volume_z_{window}"] * group["ret_1"].abs()
        group[f"signed_dollar_volume_shock_{window}"] = group[f"dollar_volume_z_{window}"] * np.sign(group["ret_1"])
    group["log_volume_z_abs_ret_z_20"] = group["log_volume_z_20"] * group["ret_1_z_20"].abs()
    group["log_volume_z_abs_ret_z_60"] = group["log_volume_z_60"] * group["ret_1_z_60"].abs()

    if use_technical_indicators:
        group = group.copy()
        group["rsi_14"] = _rsi(close, period=14)
        group["rsi_14_centered"] = (group["rsi_14"] - 50.0) / 50.0
        group["atr_14"] = _atr(high, low, close, period=14)
        macd, macd_signal, macd_hist = _macd(close)
        group["macd"] = macd
        group["macd_signal"] = macd_signal
        group["macd_hist"] = macd_hist
        group["atr_14_over_close"] = _safe_divide(group["atr_14"], close)
        group["macd_over_close"] = _safe_divide(macd, close)
        group["macd_signal_over_close"] = _safe_divide(macd_signal, close)
        group["macd_hist_over_close"] = _safe_divide(macd_hist, close)
        group["macd_hist_over_close_z_20"] = _rolling_zscore(group["macd_hist_over_close"], 20)
        group["macd_hist_over_close_z_60"] = _rolling_zscore(group["macd_hist_over_close"], 60)
        group["high_low_range_over_atr_14"] = _safe_divide(
            group["high_low_range"],
            group["atr_14_over_close"],
        )
        group["bollinger_position_20"] = _bollinger_position(close, window=20)
        group["bollinger_position_20_centered"] = group["bollinger_position_20"] - 0.5

    return group.replace([np.inf, -np.inf], np.nan)


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0.0, np.nan)


def _add_cross_sectional_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    by_date = out.groupby("date", sort=False)

    ret = out["ret_1"]
    ret_count = by_date["ret_1"].transform("count")
    ret_sum = by_date["ret_1"].transform("sum")
    current_ret_count = ret.notna().astype(int)
    other_count = ret_count - current_ret_count
    other_sum = ret_sum - ret.fillna(0.0)
    market_ret_ex_ticker = other_sum / other_count.replace(0, np.nan)
    market_ret_including_ticker = by_date["ret_1"].transform("mean")
    out["market_ret_equal_weight"] = market_ret_ex_ticker.where(
        other_count > 0,
        market_ret_including_ticker,
    )

    positive_count = by_date["ret_1"].transform(lambda series: (series > 0.0).sum())
    out["market_breadth_positive"] = positive_count / ret_count.replace(0, np.nan)
    out["ret_minus_market"] = out["ret_1"] - out["market_ret_equal_weight"]
    out["cs_ret_rank"] = by_date["ret_1"].rank(method="average")
    out["cs_ret_pct_rank"] = by_date["ret_1"].rank(method="average", pct=True)

    for window in LIQUIDITY_WINDOWS:
        shock_col = f"log_volume_z_{window}"
        out[f"cs_volume_shock_rank_{window}"] = by_date[shock_col].rank(method="average")
        out[f"cs_volume_shock_pct_rank_{window}"] = by_date[shock_col].rank(method="average", pct=True)

    return out


def _add_market_residual_features(df: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for _, group in df.groupby("ticker", sort=False, group_keys=False):
        group = group.sort_values("date").copy()
        market_ret = group["market_ret_equal_weight"]
        for beta_window in BETA_WINDOWS:
            beta_col = f"market_beta_{beta_window}"
            residual_col = f"residual_return_beta_{beta_window}"
            group[beta_col] = _rolling_beta(group["ret_1"], market_ret, window=beta_window)
            group[residual_col] = group["ret_1"] - group[beta_col] * market_ret
            for momentum_window in RESIDUAL_MOMENTUM_WINDOWS:
                group[f"residual_momentum_{momentum_window}_beta_{beta_window}"] = (
                    group[residual_col]
                    .rolling(window=momentum_window, min_periods=1)
                    .sum()
                )

        group["residual_return"] = group["residual_return_beta_20"]
        for momentum_window in RESIDUAL_MOMENTUM_WINDOWS:
            group[f"residual_momentum_{momentum_window}"] = group[
                f"residual_momentum_{momentum_window}_beta_20"
            ]
        pieces.append(group)

    return pd.concat(pieces, ignore_index=True)


def _rolling_beta(returns: pd.Series, market_returns: pd.Series, window: int) -> pd.Series:
    rolling_covariance = returns.rolling(window=window, min_periods=window).cov(market_returns)
    rolling_market_variance = market_returns.rolling(window=window, min_periods=window).var()
    return _safe_divide(rolling_covariance, rolling_market_variance)


def _rolling_percentile_rank(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=1).apply(_last_percentile_rank, raw=True)


def _last_percentile_rank(values: np.ndarray) -> float:
    current = values[-1]
    if np.isnan(current):
        return np.nan
    valid = values[~np.isnan(values)]
    if len(valid) == 0:
        return np.nan
    return float((valid <= current).mean())


def _rolling_max_drawdown(close: pd.Series, window: int) -> pd.Series:
    return close.rolling(window=window, min_periods=window).apply(_max_drawdown, raw=True)


def _max_drawdown(values: np.ndarray) -> float:
    valid = values[~np.isnan(values)]
    if len(valid) == 0:
        return np.nan
    peaks = np.maximum.accumulate(valid)
    drawdowns = valid / np.where(peaks == 0.0, np.nan, peaks) - 1.0
    finite_drawdowns = drawdowns[~np.isnan(drawdowns)]
    if len(finite_drawdowns) == 0:
        return np.nan
    return float(finite_drawdowns.min())


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    rolling = series.rolling(window=window, min_periods=window)
    return _safe_divide(series - rolling.mean(), rolling.std())


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def _macd(close: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema_12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema_26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    macd = ema_12 - ema_26
    signal = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    return macd, signal, macd - signal


def _bollinger_position(close: pd.Series, window: int = 20) -> pd.Series:
    mean = close.rolling(window=window, min_periods=window).mean()
    std = close.rolling(window=window, min_periods=window).std()
    upper = mean + 2.0 * std
    lower = mean - 2.0 * std
    return (close - lower) / (upper - lower)
