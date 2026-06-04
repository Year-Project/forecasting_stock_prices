from __future__ import annotations

import pandas as pd


def generate_walk_forward_splits(
    df: pd.DataFrame,
    date_col: str = "date",
    train_window: int = 756,
    validation_window: int = 126,
    step: int = 126,
    mode: str = "rolling",
) -> list[dict[str, pd.DatetimeIndex]]:
    if mode not in {"rolling", "expanding"}:
        raise ValueError("mode must be 'rolling' or 'expanding'")
    if min(train_window, validation_window, step) <= 0:
        raise ValueError("train_window, validation_window and step must be positive")

    dates = pd.DatetimeIndex(pd.Series(df[date_col].dropna().unique()).sort_values())
    splits: list[dict[str, pd.DatetimeIndex]] = []
    start = train_window

    while start + validation_window <= len(dates):
        train_start = 0 if mode == "expanding" else start - train_window
        train_dates = dates[train_start:start]
        validation_dates = dates[start : start + validation_window]
        if len(train_dates) and len(validation_dates):
            if train_dates.max() >= validation_dates.min():
                raise AssertionError("Train dates must be strictly before validation dates")
            splits.append({"train_dates": train_dates, "validation_dates": validation_dates})
        start += step

    return splits
