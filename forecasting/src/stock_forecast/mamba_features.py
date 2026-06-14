from __future__ import annotations

import numpy as np
import pandas as pd

MAMBA_FEATURE_PREFIX = "mamba_"
MAMBA_WINDOWS = (5, 20, 60, 126)
TRADING_DAYS_PER_YEAR = 252.0


def make_mamba_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add deterministic, leakage-safe sequence features for Mamba models."""
    required = {"date", "ticker", "open", "high", "low", "close", "volume"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing columns required for Mamba features: {missing}")

    out = df.sort_values(["ticker", "date"]).copy()
    out["date"] = pd.to_datetime(out["date"])
    pieces = []
    for _, group in out.groupby("ticker", sort=False, group_keys=False):
        pieces.append(_add_ticker_mamba_features(group.copy()))
    return pd.concat(pieces, ignore_index=True).replace([np.inf, -np.inf], np.nan)


def get_mamba_feature_columns(df: pd.DataFrame, base_feature_cols: list[str] | None = None) -> list[str]:
    """Return base numeric features plus Mamba-specific numeric additions."""
    base = [col for col in list(base_feature_cols or []) if col in df.columns]
    mamba_cols = [
        col
        for col in df.columns
        if col.startswith(MAMBA_FEATURE_PREFIX) and pd.api.types.is_numeric_dtype(df[col])
    ]
    seen = set()
    ordered = []
    for col in [*base, *mamba_cols]:
        if col not in seen:
            ordered.append(col)
            seen.add(col)
    return ordered


def mamba_feature_family_counts(feature_cols: list[str]) -> dict[str, int]:
    """Summarize Mamba feature families for EDA payloads."""
    family_rules = {
        "gap_intraday": ("gap", "intraday", "overnight"),
        "return_regime": ("hit_rate", "sign", "trend", "mean_to_vol", "drawdown"),
        "volatility_shape": ("vol", "skew", "kurt", "tail", "range"),
        "volume_imbalance": ("volume", "dollar"),
    }
    counts = {family: 0 for family in family_rules}
    counts["other_mamba"] = 0
    for col in feature_cols:
        if not col.startswith(MAMBA_FEATURE_PREFIX):
            continue
        matched = False
        for family, tokens in family_rules.items():
            if any(token in col for token in tokens):
                counts[family] += 1
                matched = True
                break
        if not matched:
            counts["other_mamba"] += 1
    return counts


def mamba_sequence_diagnostics(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    lookbacks: list[int] | None = None,
) -> dict[str, object]:
    """Summarize sequence feasibility, missingness, and feature families."""
    from .lstm_features import sequence_diagnostics

    diagnostics = sequence_diagnostics(
        df,
        feature_cols=feature_cols,
        target_col=target_col,
        lookbacks=lookbacks or [60, 90, 126, 252],
    )
    diagnostics["mamba_feature_family_counts"] = mamba_feature_family_counts(feature_cols)
    diagnostics["ticker_coverage"] = _ticker_coverage(df, target_col)
    return diagnostics


def stationary_mamba_feature_columns(feature_cols: list[str]) -> list[str]:
    """Drop level-like columns from the Mamba feature set for pooled global training."""
    level_like = {
        "log_close",
        "dollar_volume",
        "lstm_dollar_volume_log",
    }
    return [
        col
        for col in feature_cols
        if col not in level_like
        and not col.endswith("_log")
        and "dollar_volume" not in col
    ]


def _add_ticker_mamba_features(group: pd.DataFrame) -> pd.DataFrame:
    open_ = group["open"].astype(float)
    high = group["high"].astype(float)
    low = group["low"].astype(float)
    close = group["close"].astype(float)
    volume = group["volume"].astype(float)
    prev_close = close.shift(1)

    close_den = close.replace(0.0, np.nan)
    prev_close_den = prev_close.replace(0.0, np.nan)
    open_den = open_.replace(0.0, np.nan)
    low_den = low.replace(0.0, np.nan)

    ret = np.log(close / prev_close_den)
    gap_ret = np.log(open_ / prev_close_den)
    intraday_ret = np.log(close / open_den)
    high_low_ret = np.log(high / low_den)

    group["mamba_close_ret_1"] = ret
    group["mamba_gap_ret"] = gap_ret
    group["mamba_intraday_ret"] = intraday_ret
    group["mamba_overnight_abs_ret"] = gap_ret.abs()
    group["mamba_intraday_abs_ret"] = intraday_ret.abs()
    group["mamba_high_low_log_range"] = high_low_ret
    group["mamba_open_to_high"] = high / open_den - 1.0
    group["mamba_open_to_low"] = open_ / low_den - 1.0
    group["mamba_close_to_mid_range"] = _safe_divide(close - (high + low) / 2.0, high - low)
    group["mamba_gap_minus_intraday"] = gap_ret - intraday_ret
    group["mamba_gap_same_direction_intraday"] = (np.sign(gap_ret) == np.sign(intraday_ret)).astype(float)

    log_volume = np.log(volume.replace(0.0, np.nan))
    dollar_volume = close * volume
    ret_sign = np.sign(ret)
    negative_ret = ret.where(ret < 0.0, 0.0)
    positive_ret = ret.where(ret > 0.0, 0.0)
    abs_ret = ret.abs()

    for window in MAMBA_WINDOWS:
        ret_roll = ret.rolling(window=window, min_periods=window)
        gap_roll = gap_ret.rolling(window=window, min_periods=window)
        intraday_roll = intraday_ret.rolling(window=window, min_periods=window)
        abs_roll = abs_ret.rolling(window=window, min_periods=window)
        volume_roll = volume.rolling(window=window, min_periods=window)
        log_volume_roll = log_volume.rolling(window=window, min_periods=window)
        dollar_volume_roll = dollar_volume.rolling(window=window, min_periods=window)
        range_roll = high_low_ret.rolling(window=window, min_periods=window)

        realized_vol = ret_roll.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        downside_vol = negative_ret.rolling(window=window, min_periods=window).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        upside_vol = positive_ret.rolling(window=window, min_periods=window).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        gap_vol = gap_roll.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        intraday_vol = intraday_roll.std() * np.sqrt(TRADING_DAYS_PER_YEAR)

        group[f"mamba_positive_hit_rate_{window}"] = (ret > 0.0).astype(float).rolling(window, min_periods=window).mean()
        group[f"mamba_negative_hit_rate_{window}"] = (ret < 0.0).astype(float).rolling(window, min_periods=window).mean()
        group[f"mamba_sign_persistence_{window}"] = (ret_sign * ret_sign.shift(1)).rolling(window, min_periods=window).mean()
        group[f"mamba_return_trend_{window}"] = ret_roll.mean()
        group[f"mamba_return_mean_to_vol_{window}"] = _safe_divide(ret_roll.mean(), ret_roll.std())
        group[f"mamba_realized_vol_{window}"] = realized_vol
        group[f"mamba_downside_vol_{window}"] = downside_vol
        group[f"mamba_upside_downside_vol_ratio_{window}"] = _safe_divide(upside_vol, downside_vol)
        group[f"mamba_gap_vol_{window}"] = gap_vol
        group[f"mamba_intraday_vol_{window}"] = intraday_vol
        group[f"mamba_overnight_vol_share_{window}"] = _safe_divide(gap_vol, gap_vol + intraday_vol)
        group[f"mamba_realized_skew_{window}"] = ret_roll.skew()
        group[f"mamba_realized_kurt_{window}"] = ret_roll.kurt()
        group[f"mamba_abs_ret_z_{window}"] = _safe_divide(abs_ret - abs_roll.mean(), abs_roll.std())
        group[f"mamba_range_vol_{window}"] = range_roll.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        group[f"mamba_range_mean_to_vol_{window}"] = _safe_divide(range_roll.mean(), range_roll.std())
        group[f"mamba_log_volume_z_{window}"] = _safe_divide(log_volume - log_volume_roll.mean(), log_volume_roll.std())
        group[f"mamba_dollar_volume_z_{window}"] = _safe_divide(dollar_volume - dollar_volume_roll.mean(), dollar_volume_roll.std())
        group[f"mamba_signed_volume_imbalance_{window}"] = _safe_divide((ret_sign * volume).rolling(window, min_periods=window).sum(), volume_roll.sum())
        group[f"mamba_signed_dollar_volume_imbalance_{window}"] = _safe_divide(
            (ret_sign * dollar_volume).rolling(window, min_periods=window).sum(),
            dollar_volume_roll.sum(),
        )
        group[f"mamba_volume_return_corr_{window}"] = ret.rolling(window, min_periods=window).corr(log_volume)
        group[f"mamba_volume_abs_return_corr_{window}"] = abs_ret.rolling(window, min_periods=window).corr(log_volume)
        group[f"mamba_drawdown_from_high_{window}"] = close / close.rolling(window, min_periods=window).max() - 1.0

    group["mamba_vol_5_over_20"] = _safe_divide(group["mamba_realized_vol_5"], group["mamba_realized_vol_20"])
    group["mamba_vol_20_over_60"] = _safe_divide(group["mamba_realized_vol_20"], group["mamba_realized_vol_60"])
    group["mamba_vol_60_over_126"] = _safe_divide(group["mamba_realized_vol_60"], group["mamba_realized_vol_126"])
    group["mamba_gap_vol_5_over_20"] = _safe_divide(group["mamba_gap_vol_5"], group["mamba_gap_vol_20"])
    group["mamba_intraday_vol_5_over_20"] = _safe_divide(group["mamba_intraday_vol_5"], group["mamba_intraday_vol_20"])
    return group


def _ticker_coverage(df: pd.DataFrame, target_col: str) -> list[dict[str, object]]:
    rows = []
    clean = df.dropna(subset=[target_col]).copy() if target_col in df.columns else df.copy()
    for ticker, group in clean.groupby("ticker", sort=True):
        rows.append(
            {
                "ticker": str(ticker),
                "rows": int(len(group)),
                "date_start": pd.to_datetime(group["date"]).min(),
                "date_end": pd.to_datetime(group["date"]).max(),
            }
        )
    return rows


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0.0, np.nan)
