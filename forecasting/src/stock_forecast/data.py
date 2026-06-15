from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

STANDARD_COLUMNS = ["date", "ticker", "open", "high", "low", "close", "volume"]


def load_ohlcv(path: str | Path, config: dict[str, Any] | None = None) -> pd.DataFrame:
    """Load and normalize OHLCV data to date/ticker/open/high/low/close/volume."""
    cfg = config or {}
    raw = _read_table(Path(path), cfg)

    date_col = cfg.get("date_col", "date")
    ticker_col = cfg.get("ticker_col", "ticker")
    column_map = {
        date_col: "date",
        cfg.get("open_col", "open"): "open",
        cfg.get("high_col", "high"): "high",
        cfg.get("low_col", "low"): "low",
        cfg.get("close_col", "close"): "close",
        cfg.get("volume_col", "volume"): "volume",
    }
    if ticker_col in raw.columns:
        column_map[ticker_col] = "ticker"

    missing = [src for src in column_map if src not in raw.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = raw.rename(columns=column_map).copy()
    if "ticker" not in df.columns:
        df["ticker"] = "SINGLE"
    df = df[STANDARD_COLUMNS]

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if df["date"].isna().any():
        raise ValueError("Date column contains unparsable values")
    if df["close"].isna().any():
        raise ValueError("Close column contains missing values")
    if (df[["open", "high", "low", "close"]] < 0).any().any():
        raise ValueError("OHLC columns contain negative prices")
    if (df["close"] <= 0).any():
        raise ValueError("Close column must be positive")
    if np.isinf(df[["open", "high", "low", "close", "volume"]].to_numpy()).any():
        raise ValueError("OHLCV columns contain infinite values")

    df = (
        df.drop_duplicates(["ticker", "date"], keep="last")
        .sort_values(["ticker", "date"])
        .reset_index(drop=True)
    )
    return df


def _read_table(path: Path, cfg: dict[str, Any]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input data file not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path, sep=cfg.get("sep", ","))
    raise ValueError(f"Unsupported input format: {path.suffix}")
