from __future__ import annotations

import numpy as np
import pandas as pd


def make_future_return_target(
    df: pd.DataFrame,
    horizon: int,
    execution_timing: str = "close_to_close",
) -> pd.DataFrame:
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if execution_timing not in {"close_to_close", "next_open"}:
        raise ValueError("execution_timing must be one of: 'close_to_close', 'next_open'")

    out = df.sort_values(["ticker", "date"]).copy()
    grouped = out.groupby("ticker", sort=False)
    if execution_timing == "close_to_close":
        future_close = grouped["close"].shift(-horizon)
        target_date = grouped["date"].shift(-horizon)
        out[f"target_return_{horizon}"] = np.log(future_close / out["close"])
        out["future_close"] = future_close
        out["target_date"] = target_date
        return out

    entry_open = grouped["open"].shift(-1)
    future_open = grouped["open"].shift(-(horizon + 1))
    out["entry_date"] = grouped["date"].shift(-1)
    out["entry_open"] = entry_open
    out["future_open"] = future_open
    out["target_date"] = grouped["date"].shift(-(horizon + 1))
    out[f"target_return_{horizon}_next_open"] = np.log(future_open / entry_open)
    return out
