from __future__ import annotations

import math

import numpy as np
import pandas as pd


LSTM_FEATURE_PREFIX = "lstm_"
LSTM_VOLATILITY_WINDOWS = (5, 20, 60)


def make_lstm_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add deterministic, leakage-safe features useful for sequence models."""
    required = {"date", "ticker", "open", "high", "low", "close", "volume"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing columns required for LSTM features: {missing}")

    out = df.sort_values(["ticker", "date"]).copy()
    out["date"] = pd.to_datetime(out["date"])
    out = _add_calendar_features(out)
    pieces = []
    for _, group in out.groupby("ticker", sort=False, group_keys=False):
        pieces.append(_add_ticker_lstm_features(group.copy()))
    out = pd.concat(pieces, ignore_index=True)
    return out.replace([np.inf, -np.inf], np.nan)


def get_lstm_feature_columns(df: pd.DataFrame, base_feature_cols: list[str] | None = None) -> list[str]:
    """Return base numeric features plus LSTM-specific numeric additions."""
    base = [col for col in list(base_feature_cols or []) if col in df.columns]
    lstm_cols = [
        col
        for col in df.columns
        if col.startswith(LSTM_FEATURE_PREFIX) and pd.api.types.is_numeric_dtype(df[col])
    ]
    seen = set()
    ordered = []
    for col in [*base, *lstm_cols]:
        if col not in seen:
            ordered.append(col)
            seen.add(col)
    return ordered


def sequence_diagnostics(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    lookbacks: list[int] | None = None,
) -> dict[str, object]:
    """Summarize sequence feasibility and missingness for LSTM training."""
    lookbacks = lookbacks or [20, 40, 60, 90, 126]
    clean = df.sort_values(["ticker", "date"]).copy()
    rows = []
    for ticker, group in clean.dropna(subset=[target_col]).groupby("ticker", sort=True):
        n_rows = int(len(group))
        rows.append(
            {
                "ticker": ticker,
                "rows": n_rows,
                "date_start": group["date"].min(),
                "date_end": group["date"].max(),
                "max_feasible_lookback": max([lb for lb in lookbacks if n_rows >= lb] or [0]),
                **{f"sequences_lookback_{lb}": max(n_rows - lb + 1, 0) for lb in lookbacks},
            }
        )
    feature_frame = clean[feature_cols] if feature_cols else pd.DataFrame(index=clean.index)
    missingness = (
        feature_frame.isna().mean().sort_values(ascending=False).head(30).to_dict()
        if not feature_frame.empty
        else {}
    )
    target = clean[target_col].replace([np.inf, -np.inf], np.nan).dropna()
    return {
        "rows_by_ticker": rows,
        "lookbacks": lookbacks,
        "feature_count": len(feature_cols),
        "top_feature_missingness": {key: float(value) for key, value in missingness.items()},
        "target_distribution": {
            "count": int(target.count()),
            "mean": float(target.mean()) if not target.empty else float("nan"),
            "std": float(target.std()) if len(target) > 1 else float("nan"),
            "min": float(target.min()) if not target.empty else float("nan"),
            "p25": float(target.quantile(0.25)) if not target.empty else float("nan"),
            "median": float(target.median()) if not target.empty else float("nan"),
            "p75": float(target.quantile(0.75)) if not target.empty else float("nan"),
            "max": float(target.max()) if not target.empty else float("nan"),
        },
        "recommended_scaler": "median imputation + RobustScaler fit inside each strict train split",
    }


def _add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    day_of_week = df["date"].dt.dayofweek.astype(float)
    month = df["date"].dt.month.astype(float)
    day_of_year = df["date"].dt.dayofyear.astype(float)
    df["lstm_dow_sin"] = np.sin(2.0 * np.pi * day_of_week / 5.0)
    df["lstm_dow_cos"] = np.cos(2.0 * np.pi * day_of_week / 5.0)
    df["lstm_month_sin"] = np.sin(2.0 * np.pi * month / 12.0)
    df["lstm_month_cos"] = np.cos(2.0 * np.pi * month / 12.0)
    df["lstm_day_of_year_sin"] = np.sin(2.0 * np.pi * day_of_year / 366.0)
    df["lstm_day_of_year_cos"] = np.cos(2.0 * np.pi * day_of_year / 366.0)
    return df


def _add_ticker_lstm_features(group: pd.DataFrame) -> pd.DataFrame:
    open_ = group["open"].astype(float)
    high = group["high"].astype(float)
    low = group["low"].astype(float)
    close = group["close"].astype(float)
    volume = group["volume"].astype(float)

    price_den = close.replace(0.0, np.nan)
    high_low_den = (high - low).replace(0.0, np.nan)
    group["lstm_candle_body"] = (close - open_) / price_den
    group["lstm_candle_body_abs"] = group["lstm_candle_body"].abs()
    group["lstm_upper_wick"] = (high - np.maximum(open_, close)) / price_den
    group["lstm_lower_wick"] = (np.minimum(open_, close) - low) / price_den
    group["lstm_close_position"] = (close - low) / high_low_den
    group["lstm_log_high_low_range"] = np.log(high / low.replace(0.0, np.nan))
    group["lstm_log_open_close"] = np.log(close / open_.replace(0.0, np.nan))

    nonzero_volume = volume.replace(0.0, np.nan)
    log_volume = np.log(nonzero_volume)
    group["lstm_log_volume_change_1"] = log_volume.diff()
    group["lstm_dollar_volume_log"] = np.log((close * volume).replace(0.0, np.nan))
    for window in LSTM_VOLATILITY_WINDOWS:
        group[f"lstm_volume_over_sma_{window}"] = volume / volume.rolling(window, min_periods=window).mean() - 1.0
        group[f"lstm_log_volume_z_{window}"] = _rolling_zscore(log_volume, window)

    parkinson_vols = {}
    gk_vols = {}
    log_hl = np.log(high / low.replace(0.0, np.nan))
    log_co = np.log(close / open_.replace(0.0, np.nan))
    parkinson_var = (1.0 / (4.0 * math.log(2.0))) * log_hl.pow(2)
    garman_klass_var = 0.5 * log_hl.pow(2) - (2.0 * math.log(2.0) - 1.0) * log_co.pow(2)
    for window in LSTM_VOLATILITY_WINDOWS:
        parkinson = np.sqrt(TRADING_DAYS_PER_YEAR * parkinson_var.rolling(window, min_periods=window).mean())
        garman_klass = np.sqrt(
            np.maximum(TRADING_DAYS_PER_YEAR * garman_klass_var.rolling(window, min_periods=window).mean(), 0.0)
        )
        parkinson_vols[window] = parkinson
        gk_vols[window] = garman_klass
        group[f"lstm_parkinson_vol_{window}"] = parkinson
        group[f"lstm_garman_klass_vol_{window}"] = garman_klass

    group["lstm_parkinson_vol_5_over_20"] = _safe_divide(parkinson_vols[5], parkinson_vols[20])
    group["lstm_parkinson_vol_20_over_60"] = _safe_divide(parkinson_vols[20], parkinson_vols[60])
    group["lstm_garman_klass_vol_5_over_20"] = _safe_divide(gk_vols[5], gk_vols[20])
    group["lstm_garman_klass_vol_20_over_60"] = _safe_divide(gk_vols[20], gk_vols[60])
    group["lstm_volume_z_5_minus_20"] = group["lstm_log_volume_z_5"] - group["lstm_log_volume_z_20"]
    group["lstm_volume_z_20_minus_60"] = group["lstm_log_volume_z_20"] - group["lstm_log_volume_z_60"]
    return group


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0.0, np.nan)


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    rolling = series.rolling(window, min_periods=window)
    return _safe_divide(series - rolling.mean(), rolling.std())


TRADING_DAYS_PER_YEAR = 252.0
