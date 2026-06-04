from __future__ import annotations

from pathlib import Path
import re

import matplotlib.pyplot as plt
import pandas as pd

from .utils import ensure_dir


def save_per_ticker_plots(
    predictions: pd.DataFrame,
    output_dir: str | Path,
    equity_curve: pd.DataFrame | None = None,
    max_tickers: int | None = None,
) -> None:
    out = ensure_dir(Path(output_dir) / "tickers")
    if predictions.empty:
        return

    tickers = sorted(predictions["ticker"].dropna().astype(str).unique())
    if max_tickers is not None:
        tickers = tickers[:max_tickers]

    for ticker in tickers:
        ticker_predictions = predictions[predictions["ticker"].astype(str) == ticker].sort_values("date")
        safe_ticker = _safe_filename(ticker)
        _ticker_prediction_plot(
            ticker_predictions,
            out / f"{safe_ticker}_predicted_vs_actual_by_date.png",
            f"Predicted vs actual return: {ticker}",
        )
        _ticker_scatter_plot(
            ticker_predictions,
            out / f"{safe_ticker}_predicted_vs_actual_scatter.png",
            f"Predicted vs actual scatter: {ticker}",
        )
        if equity_curve is not None and not equity_curve.empty:
            ticker_equity = equity_curve[equity_curve["ticker"].astype(str) == ticker].sort_values("date")
            if not ticker_equity.empty:
                _ticker_equity_plot(
                    ticker_equity,
                    out / f"{safe_ticker}_signal_equity.png",
                    f"Signal equity by model: {ticker}",
                )


def save_feature_importance(model, feature_names: list[str], output_path: str | Path) -> None:
    estimator = model.named_steps.get("model", model) if hasattr(model, "named_steps") else model
    values = getattr(estimator, "feature_importances_", None)
    if values is None and hasattr(estimator, "coef_"):
        values = abs(estimator.coef_)
    if values is None:
        return
    importance = pd.DataFrame({"feature": feature_names, "importance": values}).sort_values(
        "importance", ascending=False
    )
    top = importance.head(40).sort_values("importance")
    plt.figure(figsize=(10, max(5, len(top) * 0.25)))
    plt.barh(top["feature"], top["importance"])
    plt.title("Feature importance")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _ticker_prediction_plot(df: pd.DataFrame, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    actual = df.drop_duplicates(["date", "ticker"]).sort_values("date")
    ax.plot(actual["date"], actual["y_true"], color="black", linewidth=1.7, label="actual")
    for model_name, group in df.groupby("model_name"):
        ax.plot(group["date"], group["y_pred"], linewidth=1.1, alpha=0.85, label=model_name)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(title)
    ax.set_ylabel("Return")
    ax.legend(ncol=min(df["model_name"].nunique() + 1, 4))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _ticker_scatter_plot(df: pd.DataFrame, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    for model_name, group in df.groupby("model_name"):
        ax.scatter(group["y_true"], group["y_pred"], s=8, alpha=0.35, label=model_name)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Actual return")
    ax.set_ylabel("Predicted return")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _ticker_equity_plot(df: pd.DataFrame, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    for model_name, group in df.groupby("model_name"):
        ax.plot(group["date"], group["cumulative_return"], linewidth=1.3, label=model_name)
    ax.set_title(title)
    ax.set_ylabel("Cumulative return")
    ax.legend(ncol=min(df["model_name"].nunique(), 4))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "ticker"
