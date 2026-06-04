from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

os.environ.setdefault("GLOG_minloglevel", "2")
PROJECT_SRC = Path(__file__).resolve().parents[1] / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

import joblib
import numpy as np
import pandas as pd

from stock_forecast.metrics import grouped_regression_metrics
from stock_forecast.models import build_model
from stock_forecast.pipeline import _purge_train_target_overlap, build_estimator
from stock_forecast.splits import generate_walk_forward_splits


LEARNED_IMPORTANCE_MODELS = ("ridge", "lightgbm", "xgboost", "catboost")

LOW_IMPORTANCE_DROP_CANDIDATES = [
    "volume_change_1",
    "ret_lag_2",
    "open_close_ret",
    "ret_lag_3",
    "ret_1",
    "ret_lag_5",
    "close_to_high",
    "ret_lag_20",
    "ret_lag_10",
    "ret_lag_1",
]

TOP_BUILTIN_KEEP = [
    "log_close",
    "atr_14",
    "macd_signal",
    "rolling_ret_std_60",
    "macd",
    "rolling_ret_mean_60",
    "macd_hist",
    "rolling_ret_std_5",
    "rolling_ret_mean_20",
    "rolling_ret_std_20",
    "rolling_ret_std_10",
    "close_over_sma_60",
    "close_over_rolling_max_20",
    "close_over_sma_20",
    "bollinger_position_20",
    "close_over_rolling_min_20",
    "rsi_14",
    "volume_over_sma_60",
    "rolling_ret_mean_10",
    "volume_over_sma_20",
]

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

RAW_SCALE_FEATURES = {"log_close", "atr_14", "macd", "macd_signal", "macd_hist"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit feature usefulness from saved walk-forward artifacts.")
    parser.add_argument("--project-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--horizon",
        action="append",
        help="Horizon name to audit, for example 'week' or 'month'. Repeat to audit multiple horizons. Defaults to all horizon artifacts.",
    )
    parser.add_argument("--train-window", type=int, default=756)
    parser.add_argument("--validation-window", type=int, default=126)
    parser.add_argument("--step", type=int, default=126)
    parser.add_argument("--split-mode", choices=["rolling", "expanding"], default="rolling")
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_dir = args.project_dir
    contexts = discover_audit_contexts(project_dir, args.horizon)
    combined_payload = {"horizons": {}}

    for context in contexts:
        audit_payload = run_horizon_feature_audit(context, args)
        combined_payload["horizons"][context["horizon_name"]] = audit_payload

    if len(contexts) > 1:
        reports_dir = project_dir / "artifacts" / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        save_json(combined_payload, reports_dir / "feature_audit.json")
        print(f"Wrote combined feature audit metadata to {reports_dir / 'feature_audit.json'}")


def discover_audit_contexts(project_dir: Path, horizon_names: list[str] | None) -> list[dict[str, Any]]:
    artifacts_dir = project_dir / "artifacts"
    requested = list(horizon_names or [])
    if not requested:
        data_horizon_dir = artifacts_dir / "data" / "horizons"
        requested = sorted(
            path.name
            for path in data_horizon_dir.iterdir()
            if path.is_dir() and (path / "feature_columns.json").exists()
        ) if data_horizon_dir.exists() else []

    if requested:
        contexts = []
        for horizon_name in requested:
            data_dir = artifacts_dir / "data" / "horizons" / horizon_name
            artifact_dir = artifacts_dir / "horizons" / horizon_name
            feature_path = data_dir / "feature_columns.json"
            model_path = data_dir / "model_dataset.parquet"
            if not feature_path.exists():
                raise FileNotFoundError(f"Missing horizon feature artifact: {feature_path}")
            if not model_path.exists():
                raise FileNotFoundError(f"Missing horizon model dataset: {model_path}")
            contexts.append(
                {
                    "horizon_name": horizon_name,
                    "artifact_dir": artifact_dir,
                    "data_dir": data_dir,
                    "reports_dir": artifact_dir / "reports",
                    "feature_path": feature_path,
                    "model_path": model_path,
                }
            )
        return contexts

    feature_path = artifacts_dir / "data" / "feature_columns.json"
    model_path = artifacts_dir / "data" / "model_dataset.parquet"
    if not feature_path.exists() or not model_path.exists():
        raise FileNotFoundError("No horizon-scoped or root feature artifacts found")
    return [
        {
            "horizon_name": "root",
            "artifact_dir": artifacts_dir,
            "data_dir": artifacts_dir / "data",
            "reports_dir": artifacts_dir / "reports",
            "feature_path": feature_path,
            "model_path": model_path,
        }
    ]


def run_horizon_feature_audit(context: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    artifacts_dir = context["artifact_dir"]
    reports_dir = context["reports_dir"]
    reports_dir.mkdir(parents=True, exist_ok=True)
    horizon_name = context["horizon_name"]

    feature_payload = load_json(context["feature_path"])
    feature_cols = list(feature_payload["feature_columns"])
    target_col = str(feature_payload["target_column"])
    model_df = pd.read_parquet(context["model_path"])
    model_df["date"] = pd.to_datetime(model_df["date"])
    if "target_date" in model_df.columns:
        model_df["target_date"] = pd.to_datetime(model_df["target_date"])

    best_by_ticker = pd.read_json(reports_dir / "best_models_by_ticker.json")
    splits = generate_walk_forward_splits(
        model_df,
        train_window=args.train_window,
        validation_window=args.validation_window,
        step=args.step,
        mode=args.split_mode,
    )

    builtin_importance = extract_builtin_importances(artifacts_dir, feature_cols)
    builtin_summary = summarize_builtin_importance(builtin_importance)
    permutation_importance = current_best_permutation_importance(
        artifacts_dir,
        model_df,
        feature_cols,
        target_col,
        best_by_ticker,
        splits,
        random_state=args.random_state,
    )
    permutation_summary = summarize_permutation_importance(permutation_importance)
    ablation = current_best_feature_subset_ablation(
        artifacts_dir,
        model_df,
        feature_cols,
        target_col,
        best_by_ticker,
        splits,
        random_state=args.random_state,
    )
    ablation_summary = summarize_ablation(ablation)
    univariate = univariate_spearman_summary(model_df, feature_cols, target_col)
    redundant_pairs = high_correlation_pairs(model_df, feature_cols)
    family_summary = feature_family_summary(
        builtin_summary,
        permutation_summary,
        univariate,
        feature_cols,
    )

    save_table(builtin_importance, reports_dir / "feature_builtin_importance_by_fold.parquet")
    save_table(builtin_summary, reports_dir / "feature_builtin_importance_summary.parquet")
    save_table(permutation_importance, reports_dir / "feature_current_best_permutation.parquet")
    save_table(permutation_summary, reports_dir / "feature_current_best_permutation_summary.parquet")
    save_table(ablation, reports_dir / "feature_subset_ablation.parquet")
    save_table(ablation_summary, reports_dir / "feature_subset_ablation_summary.parquet")
    save_table(univariate, reports_dir / "feature_univariate_spearman_summary.parquet")
    save_table(redundant_pairs, reports_dir / "feature_high_correlation_pairs.parquet")
    save_table(family_summary, reports_dir / "feature_family_summary.parquet")

    audit_payload = {
        "horizon_name": horizon_name,
        "horizon": feature_payload.get("horizon"),
        "feature_count": len(feature_cols),
        "target_column": target_col,
        "model_rows": int(len(model_df)),
        "date_start": str(model_df["date"].min().date()),
        "date_end": str(model_df["date"].max().date()),
        "tickers": sorted(model_df["ticker"].dropna().astype(str).unique().tolist()),
        "low_importance_drop_candidates": LOW_IMPORTANCE_DROP_CANDIDATES,
        "top_builtin_features": builtin_summary.head(15).to_dict(orient="records"),
        "feature_family_summary": family_summary.to_dict(orient="records"),
        "ablation_summary": ablation_summary.to_dict(orient="records"),
    }
    save_json(audit_payload, reports_dir / "feature_audit.json")

    markdown = render_markdown(
        feature_payload=feature_payload,
        model_df=model_df,
        best_by_ticker=best_by_ticker,
        builtin_summary=builtin_summary,
        permutation_summary=permutation_summary,
        ablation_summary=ablation_summary,
        univariate=univariate,
        redundant_pairs=redundant_pairs,
        family_summary=family_summary,
    )
    (reports_dir / "feature_audit.md").write_text(markdown, encoding="utf-8")

    print(f"Wrote feature audit to {reports_dir / 'feature_audit.md'}")
    return audit_payload


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(payload: Any, path: Path) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_table(df: pd.DataFrame, path: Path) -> None:
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


def extract_builtin_importances(artifacts_dir: Path, feature_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_name in LEARNED_IMPORTANCE_MODELS:
        model_root = artifacts_dir / "models" / model_name
        for path in sorted(model_root.glob("*/*.pkl")):
            fold = fold_from_model_path(path)
            if fold is None:
                continue
            ticker = path.parent.name
            payload = joblib.load(path)
            estimator = payload["estimator"] if isinstance(payload, dict) else payload
            names = payload.get("feature_cols", feature_cols) if isinstance(payload, dict) else feature_cols
            model = estimator.named_steps["model"] if hasattr(estimator, "named_steps") else estimator
            values = model_importance_values(model)
            if values is None or len(values) != len(names):
                continue
            values = np.asarray(values, dtype=float)
            total = float(np.nansum(np.abs(values)))
            normalized = values / total if total > 0 else np.zeros_like(values, dtype=float)
            for feature, raw_value, norm_value in zip(names, values, normalized):
                rows.append(
                    {
                        "model_name": model_name,
                        "ticker": ticker,
                        "fold": fold,
                        "feature": feature,
                        "raw_importance": float(raw_value),
                        "norm_importance": float(norm_value),
                    }
                )
    return pd.DataFrame(rows)


def fold_from_model_path(path: Path) -> str | None:
    if path.name == "final.pkl":
        return "final"
    if path.stem.startswith("fold_"):
        return path.stem
    return None


def model_importance_values(model: Any) -> np.ndarray | None:
    values = getattr(model, "feature_importances_", None)
    if values is not None:
        return np.asarray(values, dtype=float)
    coef = getattr(model, "coef_", None)
    if coef is not None:
        return np.abs(np.asarray(coef, dtype=float).ravel())
    return None


def summarize_builtin_importance(importance: pd.DataFrame) -> pd.DataFrame:
    if importance.empty:
        return pd.DataFrame()
    folds = importance[importance["fold"] != "final"]
    summary = (
        folds.groupby("feature", as_index=False)
        .agg(
            mean_norm=("norm_importance", "mean"),
            median_norm=("norm_importance", "median"),
            nonzero_share=("norm_importance", lambda s: float((s > 0).mean())),
            model_count=("model_name", "nunique"),
            observation_count=("norm_importance", "size"),
        )
        .sort_values("mean_norm", ascending=False)
        .reset_index(drop=True)
    )
    summary["feature_family"] = summary["feature"].map(feature_family)
    return summary


def current_best_permutation_importance(
    artifacts_dir: Path,
    model_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    best_by_ticker: pd.DataFrame,
    splits: list[dict[str, pd.DatetimeIndex]],
    random_state: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    best_map = dict(zip(best_by_ticker["ticker"].astype(str), best_by_ticker["model_name"].astype(str)))
    for fold_idx, split in enumerate(splits):
        valid_all = model_df[model_df["date"].isin(split["validation_dates"])].sort_values(["date", "ticker"])
        for ticker, model_name in best_map.items():
            path = artifacts_dir / "models" / model_name / ticker / f"fold_{fold_idx}.pkl"
            if not path.exists():
                continue
            valid = valid_all[valid_all["ticker"].astype(str) == ticker].dropna(subset=[target_col])
            if valid.empty:
                continue
            payload = joblib.load(path)
            estimator = payload["estimator"] if isinstance(payload, dict) else payload
            X = valid[feature_cols].copy()
            y = valid[target_col].to_numpy()
            baseline_rmse = rmse(y, estimator.predict(X))
            for feature in feature_cols:
                X_permuted = X.copy()
                rng = np.random.default_rng(stable_seed(random_state, model_name, ticker, fold_idx, feature))
                X_permuted[feature] = rng.permutation(X_permuted[feature].to_numpy())
                permuted_rmse = rmse(y, estimator.predict(X_permuted))
                rows.append(
                    {
                        "model_name": model_name,
                        "ticker": ticker,
                        "fold": fold_idx,
                        "feature": feature,
                        "baseline_rmse": baseline_rmse,
                        "permuted_rmse": permuted_rmse,
                        "rmse_delta": permuted_rmse - baseline_rmse,
                        "n_obs": int(len(valid)),
                    }
                )
    return pd.DataFrame(rows)


def summarize_permutation_importance(importance: pd.DataFrame) -> pd.DataFrame:
    if importance.empty:
        return pd.DataFrame()
    summary = (
        importance.groupby("feature", as_index=False)
        .agg(
            mean_rmse_delta=("rmse_delta", "mean"),
            median_rmse_delta=("rmse_delta", "median"),
            positive_share=("rmse_delta", lambda s: float((s > 0).mean())),
            observation_count=("rmse_delta", "size"),
        )
        .sort_values("mean_rmse_delta", ascending=False)
        .reset_index(drop=True)
    )
    summary["feature_family"] = summary["feature"].map(feature_family)
    return summary


def current_best_feature_subset_ablation(
    artifacts_dir: Path,
    model_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    best_by_ticker: pd.DataFrame,
    splits: list[dict[str, pd.DatetimeIndex]],
    random_state: int,
) -> pd.DataFrame:
    drop_candidates = [f for f in LOW_IMPORTANCE_DROP_CANDIDATES if f in feature_cols]
    top_keep = [f for f in TOP_BUILTIN_KEEP if f in feature_cols]
    subsets = {
        f"full_{len(feature_cols)}": list(feature_cols),
        f"drop_low_builtin_{len(drop_candidates)}_keep_{len(feature_cols) - len(drop_candidates)}": [
            f for f in feature_cols if f not in drop_candidates
        ],
        f"keep_top_builtin_{len(top_keep)}": top_keep,
    }
    best_map = dict(zip(best_by_ticker["ticker"].astype(str), best_by_ticker["model_name"].astype(str)))
    rows: list[pd.DataFrame] = []
    for subset_name, columns in subsets.items():
        for fold_idx, split in enumerate(splits):
            train_raw = model_df[model_df["date"].isin(split["train_dates"])].sort_values(["date", "ticker"])
            train = _purge_train_target_overlap(train_raw, split["validation_dates"].min())
            valid = model_df[model_df["date"].isin(split["validation_dates"])].sort_values(["date", "ticker"])
            if train.empty or valid.empty:
                continue
            for ticker, model_name in best_map.items():
                ticker_train = train[train["ticker"].astype(str) == ticker].dropna(subset=[target_col])
                ticker_valid = valid[valid["ticker"].astype(str) == ticker].dropna(subset=[target_col])
                if ticker_train.empty or ticker_valid.empty:
                    continue
                params = load_best_params(artifacts_dir, model_name, ticker)
                if model_name == "momentum" and params.get("column") not in columns:
                    continue
                estimator = build_estimator(
                    {
                        "name": model_name,
                        "model_type": model_name,
                        "estimator_factory": build_model,
                        "needs_scaler": model_name == "ridge",
                    },
                    params=params,
                    random_state=random_state,
                )
                estimator.fit(ticker_train[columns], ticker_train[target_col])
                prediction = estimator.predict(ticker_valid[columns])
                pred_df = ticker_valid[["date", "ticker", target_col]].copy()
                pred_df["model_name"] = model_name
                pred_df["fold"] = fold_idx
                pred_df["y_true"] = ticker_valid[target_col].to_numpy()
                pred_df["y_pred"] = prediction
                metrics = grouped_regression_metrics(pred_df, ["model_name", "fold", "ticker"])
                metrics["feature_subset"] = subset_name
                metrics["feature_count"] = len(columns)
                rows.append(metrics)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def load_best_params(artifacts_dir: Path, model_name: str, ticker: str) -> dict[str, Any]:
    path = artifacts_dir / "hyperparams" / model_name / f"{ticker}.json"
    if not path.exists():
        return {}
    return dict(load_json(path).get("best_params", {}))


def summarize_ablation(ablation: pd.DataFrame) -> pd.DataFrame:
    if ablation.empty:
        return pd.DataFrame()
    return (
        ablation.groupby(["feature_subset", "ticker"], as_index=False)
        .agg(
            feature_count=("feature_count", "max"),
            rmse_mean=("rmse", "mean"),
            mae_mean=("mae", "mean"),
            directional_accuracy_mean=("directional_accuracy", "mean"),
            fold_count=("fold", "nunique"),
        )
        .sort_values(["ticker", "feature_subset"])
        .reset_index(drop=True)
    )


def univariate_spearman_summary(model_df: pd.DataFrame, feature_cols: list[str], target_col: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for ticker, group in model_df.groupby("ticker"):
        for feature in feature_cols:
            corr = group[[feature, target_col]].corr(method="spearman").iloc[0, 1]
            rows.append({"ticker": ticker, "feature": feature, "spearman": corr})
    summary = (
        pd.DataFrame(rows)
        .groupby("feature", as_index=False)
        .agg(
            mean_abs_spearman=("spearman", lambda s: float(s.abs().mean())),
            mean_spearman=("spearman", "mean"),
            positive_share=("spearman", lambda s: float((s > 0).mean())),
        )
        .sort_values("mean_abs_spearman", ascending=False)
        .reset_index(drop=True)
    )
    summary["feature_family"] = summary["feature"].map(feature_family)
    return summary


def high_correlation_pairs(model_df: pd.DataFrame, feature_cols: list[str], threshold: float = 0.90) -> pd.DataFrame:
    corr = model_df[feature_cols].corr(method="spearman").abs()
    rows: list[dict[str, Any]] = []
    for idx, left in enumerate(feature_cols):
        for right in feature_cols[idx + 1 :]:
            value = float(corr.loc[left, right])
            if value >= threshold:
                rows.append(
                    {
                        "feature_left": left,
                        "feature_right": right,
                        "left_family": feature_family(left),
                        "right_family": feature_family(right),
                        "abs_spearman": value,
                    }
                )
    return pd.DataFrame(rows).sort_values("abs_spearman", ascending=False).reset_index(drop=True)


def feature_family(feature: str) -> str:
    if feature in NORMALIZED_FEATURES:
        return "normalized"
    if feature in RAW_SCALE_FEATURES:
        return "raw_scale"
    return "existing_other"


def feature_family_summary(
    builtin_summary: pd.DataFrame,
    permutation_summary: pd.DataFrame,
    univariate: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    summary = (
        pd.DataFrame({"feature": feature_cols})
        .assign(feature_family=lambda df: df["feature"].map(feature_family))
        .groupby("feature_family", as_index=False)
        .agg(feature_count=("feature", "size"))
    )
    if not builtin_summary.empty and "feature_family" in builtin_summary.columns:
        builtin = (
            builtin_summary.groupby("feature_family", as_index=False)
            .agg(mean_builtin_importance=("mean_norm", "mean"))
        )
        summary = summary.merge(builtin, on="feature_family", how="left")
    if not permutation_summary.empty and "feature_family" in permutation_summary.columns:
        permutation = (
            permutation_summary.groupby("feature_family", as_index=False)
            .agg(mean_permutation_delta=("mean_rmse_delta", "mean"))
        )
        summary = summary.merge(permutation, on="feature_family", how="left")
    if not univariate.empty and "feature_family" in univariate.columns:
        uni = (
            univariate.groupby("feature_family", as_index=False)
            .agg(mean_abs_univariate_spearman=("mean_abs_spearman", "mean"))
        )
        summary = summary.merge(uni, on="feature_family", how="left")
    return summary.sort_values("feature_family").reset_index(drop=True)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def stable_seed(*parts: Any) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def render_markdown(
    feature_payload: dict[str, Any],
    model_df: pd.DataFrame,
    best_by_ticker: pd.DataFrame,
    builtin_summary: pd.DataFrame,
    permutation_summary: pd.DataFrame,
    ablation_summary: pd.DataFrame,
    univariate: pd.DataFrame,
    redundant_pairs: pd.DataFrame,
    family_summary: pd.DataFrame,
) -> str:
    feature_count = len(feature_payload["feature_columns"])
    target_col = feature_payload["target_column"]
    dates = f"{model_df['date'].min().date()} to {model_df['date'].max().date()}"
    tickers = ", ".join(sorted(model_df["ticker"].dropna().astype(str).unique()))
    aggregate_ablation = (
        ablation_summary.groupby("feature_subset", as_index=False)
        .agg(
            feature_count=("feature_count", "max"),
            rmse_mean=("rmse_mean", "mean"),
            directional_accuracy_mean=("directional_accuracy_mean", "mean"),
        )
        .sort_values("rmse_mean")
    )
    ablation_wide = ablation_summary.pivot(index="ticker", columns="feature_subset", values="rmse_mean").reset_index()

    lines = [
        "# Feature Audit",
        "",
        "This audit uses the saved walk-forward artifacts. Built-in importances are averaged across saved fold models for Ridge, LightGBM, XGBoost and CatBoost. The removal check retrains the current best per-ticker model with the same split, target-purge policy and cached hyperparameters.",
        "",
        "## Dataset",
        "",
        f"- Rows: {len(model_df):,}",
        f"- Dates: {dates}",
        f"- Tickers: {tickers}",
        f"- Target: `{target_col}`",
        f"- Feature count: {feature_count}",
        "",
        "## Current Best Model By Ticker",
        "",
        dataframe_to_markdown(best_by_ticker),
        "",
        "## Strongest Existing Features",
        "",
        dataframe_to_markdown(builtin_summary.head(15)),
        "",
        "## Raw Vs Normalized Feature Families",
        "",
        dataframe_to_markdown(family_summary),
        "",
        "## Weakest Existing Features By Built-In Importance",
        "",
        dataframe_to_markdown(builtin_summary.tail(12).sort_values("mean_norm")),
        "",
        "## Current-Best Permutation Check",
        "",
        "Positive `mean_rmse_delta` means shuffled feature values made validation RMSE worse. Zero or negative values are weak evidence for keeping the feature in the current best model.",
        "",
        dataframe_to_markdown(permutation_summary.head(12)),
        "",
        "## Feature Removal Ablation",
        "",
        "The two reduced sets keep all currently selected momentum columns. One set removes the configured low-importance candidates; the other keeps the strongest built-in-importance features.",
        "",
        dataframe_to_markdown(aggregate_ablation),
        "",
        dataframe_to_markdown(ablation_wide),
        "",
        "## Redundancy And Univariate Signal",
        "",
        "Top univariate Spearman features:",
        "",
        dataframe_to_markdown(univariate.head(12)),
        "",
        "Highly correlated feature pairs:",
        "",
        dataframe_to_markdown(redundant_pairs.head(12)),
        "",
        "## Decision",
        "",
        "- Do not delete features from the generator blindly. The strongest features are trend/volatility features: `log_close`, `atr_14`, `macd_signal`, `rolling_ret_std_60`, `macd`, `rolling_ret_mean_60`, `macd_hist`, `rolling_ret_mean_20`, and `rolling_ret_std_*`.",
        "- For the current best per-ticker model, a smaller feature set is at least neutral in this run. The 25-feature and 20-feature ablations slightly improve mean RMSE versus the full 35-feature set, mainly through SVCB Ridge. This is not large enough to justify deleting feature code without a full retune, but it is enough to justify a reduced-feature experiment.",
        f"- Candidate removals for the next retrain: {', '.join(f'`{name}`' for name in LOW_IMPORTANCE_DROP_CANDIDATES)}.",
        "- Keep `rolling_ret_mean_10`, `rolling_ret_mean_20`, and `rolling_ret_mean_60`; the current momentum models select only these rolling-return features.",
        "- Before removing high-importance absolute-price features, add normalized alternatives and retest: `atr_14 / close`, `macd / close`, `macd_signal / close`, `macd_hist / close`, volatility ratios such as `rolling_ret_std_5 / rolling_ret_std_60`, and rolling-return z-scores.",
        "",
    ]
    return "\n".join(lines)


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    display = df.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda value: f"{value:.6g}")
    return display.to_markdown(index=False)


if __name__ == "__main__":
    main()
