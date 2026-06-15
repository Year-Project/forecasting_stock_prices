from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> dict[str, float]:
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    return {
        "mae": float(mean_absolute_error(y_true_arr, y_pred_arr)),
        "rmse": float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr))),
        "r2": float(r2_score(y_true_arr, y_pred_arr)) if len(y_true_arr) > 1 else float("nan"),
        "pearson": _safe_corr(y_true_arr, y_pred_arr, method="pearson"),
        "spearman": _safe_corr(y_true_arr, y_pred_arr, method="spearman"),
        "directional_accuracy": directional_accuracy(y_true_arr, y_pred_arr),
    }


def grouped_regression_metrics(
    predictions: pd.DataFrame,
    group_cols: str | list[str],
    target_col: str = "y_true",
    pred_col: str = "y_pred",
) -> pd.DataFrame:
    """Compute regression metrics independently for each group."""
    group_columns = [group_cols] if isinstance(group_cols, str) else list(group_cols)
    rows = []
    for group_key, group in predictions.groupby(group_columns, dropna=False):
        key_values = group_key if isinstance(group_key, tuple) else (group_key,)
        valid = group[[target_col, pred_col]].dropna()
        row = dict(zip(group_columns, key_values))
        row["n_obs"] = int(len(valid))
        if valid.empty:
            row.update({key: float("nan") for key in _REGRESSION_METRIC_KEYS})
        else:
            row.update(regression_metrics(valid[target_col], valid[pred_col]))
        rows.append(row)
    return pd.DataFrame(rows)


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not mask.any():
        return float("nan")
    return float((np.sign(y_true[mask]) == np.sign(y_pred[mask])).mean())


def _safe_corr(y_true: np.ndarray, y_pred: np.ndarray, method: str) -> float:
    series_true = pd.Series(y_true)
    series_pred = pd.Series(y_pred)
    if series_true.nunique(dropna=True) <= 1 or series_pred.nunique(dropna=True) <= 1:
        return float("nan")
    return float(series_true.corr(series_pred, method=method))


_REGRESSION_METRIC_KEYS = ["mae", "rmse", "r2", "pearson", "spearman", "directional_accuracy"]
