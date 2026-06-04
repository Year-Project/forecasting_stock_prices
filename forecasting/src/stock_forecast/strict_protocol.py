from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .artifacts import load_json, save_json, save_table
from .backtest import run_test_signal_backtest
from .metrics import grouped_regression_metrics, regression_metrics
from .pipeline import (
    METRIC_DIRECTIONS,
    TARGET_PURGE_POLICY,
    build_estimator,
    normalize_search_space,
    suggest_params,
    _ensure_target_date,
    _feature_set_metadata,
    _final_model_metadata,
    _load_matching_model_estimator,
    _purge_train_target_overlap,
    _save_model_payload,
    _target_execution_timing,
)
from .utils import ensure_dir


STRICT_PROTOCOL_DIR = "strict_protocol"


def make_strict_outer_splits(
    model_df: pd.DataFrame,
    target_col: str,
    validation_rows: int = 126,
    test_rows: int = 126,
    mature_min_rows: int = 1008,
    limited_history_min_block_rows: int = 42,
    min_train_rows: int = 60,
    max_train_rows: int | None = 1260,
) -> pd.DataFrame:
    """Create one chronological train/validation/test split per ticker."""
    if min(validation_rows, test_rows, mature_min_rows, limited_history_min_block_rows, min_train_rows) <= 0:
        raise ValueError("split row counts must be positive")
    if max_train_rows is not None and max_train_rows < min_train_rows:
        raise ValueError("max_train_rows must be None or at least min_train_rows")

    data = _ensure_target_date(model_df, target_col).dropna(subset=[target_col]).copy()
    data["date"] = pd.to_datetime(data["date"])
    data["target_date"] = pd.to_datetime(data["target_date"])
    rows = []
    for ticker, group in data.sort_values(["ticker", "date"]).groupby("ticker", sort=True):
        ticker_data = group.sort_values("date").reset_index(drop=True)
        n_obs = len(ticker_data)
        if n_obs < min_train_rows + 2:
            rows.append(
                {
                    "ticker": ticker,
                    "status": "insufficient_history",
                    "split_quality": "insufficient_history",
                    "limited_history": True,
                    "n_obs": int(n_obs),
                    "n_train_available": 0,
                    "n_train": 0,
                    "n_refit_available": 0,
                    "n_refit": 0,
                    "max_train_rows": max_train_rows,
                    "n_validation": 0,
                    "n_test": 0,
                }
            )
            continue

        if n_obs >= mature_min_rows:
            n_validation = validation_rows
            n_test = test_rows
            split_quality = "mature"
            limited_history = False
        else:
            max_block = max(1, (n_obs - min_train_rows) // 2)
            adaptive_block = max(limited_history_min_block_rows, int(round(n_obs * 0.20)))
            n_validation = min(validation_rows, adaptive_block, max_block)
            n_test = min(test_rows, adaptive_block, max_block)
            split_quality = "limited_history"
            limited_history = True

        n_train_available = n_obs - n_validation - n_test
        if n_train_available < min_train_rows or n_validation <= 0 or n_test <= 0:
            rows.append(
                {
                    "ticker": ticker,
                    "status": "insufficient_history",
                    "split_quality": "insufficient_history",
                    "limited_history": True,
                    "n_obs": int(n_obs),
                    "n_train_available": int(max(n_train_available, 0)),
                    "n_train": int(max(n_train_available, 0)),
                    "n_refit_available": int(max(n_obs - n_test, 0)),
                    "n_refit": int(max(n_obs - n_test, 0)),
                    "max_train_rows": max_train_rows,
                    "n_validation": int(max(n_validation, 0)),
                    "n_test": int(max(n_test, 0)),
                }
            )
            continue

        n_train = min(n_train_available, max_train_rows) if max_train_rows is not None else n_train_available
        n_refit_available = n_obs - n_test
        n_refit = min(n_refit_available, max_train_rows) if max_train_rows is not None else n_refit_available
        train_start_idx = n_train_available - n_train
        validation_start_idx = n_train_available
        test_start_idx = n_train_available + n_validation
        refit_start_idx = n_refit_available - n_refit

        train = ticker_data.iloc[train_start_idx:validation_start_idx]
        validation = ticker_data.iloc[validation_start_idx:test_start_idx]
        test = ticker_data.iloc[test_start_idx:]
        refit = ticker_data.iloc[refit_start_idx:test_start_idx]
        rows.append(
            {
                "ticker": ticker,
                "status": "ok",
                "split_quality": split_quality,
                "limited_history": bool(limited_history),
                "n_obs": int(n_obs),
                "n_train_available": int(n_train_available),
                "n_train": int(len(train)),
                "n_refit_available": int(n_refit_available),
                "n_refit": int(len(refit)),
                "max_train_rows": max_train_rows,
                "n_validation": int(len(validation)),
                "n_test": int(len(test)),
                "train_start": train["date"].min(),
                "train_end": train["date"].max(),
                "refit_start": refit["date"].min(),
                "refit_end": refit["date"].max(),
                "validation_start": validation["date"].min(),
                "validation_end": validation["date"].max(),
                "test_start": test["date"].min(),
                "test_end": test["date"].max(),
                "train_target_end": train["target_date"].max(),
                "refit_target_end": refit["target_date"].max(),
                "validation_target_end": validation["target_date"].max(),
                "test_target_end": test["target_date"].max(),
            }
        )
    return pd.DataFrame(rows)


def run_strict_per_ticker_protocol(
    model_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    model_configs: list[dict[str, Any]],
    artifact_dir: str | Path,
    force_retrain: bool = False,
    primary_metric: str = "directional_accuracy",
    random_state: int = 42,
    run_metadata: dict[str, Any] | None = None,
    validation_rows: int = 126,
    test_rows: int = 126,
    mature_min_rows: int = 1008,
    limited_history_min_block_rows: int = 42,
    min_train_rows: int = 60,
    max_train_rows: int | None = 1260,
    inner_max_folds: int = 5,
    inner_min_train_rows: int = 126,
    inner_validation_window: int | None = None,
    limited_history_n_trials: int = 8,
    transaction_cost_bps: float = 10.0,
    slippage_bps: float = 5.0,
    long_threshold: float = 0.0,
    signal_anchor: str = "expanding_median",
) -> dict[str, Any]:
    """Run strict per-ticker model tuning, validation ranking, final refit, and test evaluation."""
    artifact_root = Path(artifact_dir) / STRICT_PROTOCOL_DIR
    reports_dir = ensure_dir(artifact_root / "reports")
    models_dir = ensure_dir(artifact_root / "models")
    hyperparams_dir = ensure_dir(artifact_root / "hyperparams")
    run_metadata = dict(run_metadata or {})

    data = _ensure_target_date(model_df, target_col).dropna(subset=[target_col]).copy()
    data["date"] = pd.to_datetime(data["date"])
    for col in ["entry_date", "target_date"]:
        if col in data.columns:
            data[col] = pd.to_datetime(data[col])
    execution_timing = _target_execution_timing(target_col)
    horizon = int(run_metadata.get("horizon", _infer_horizon(target_col)))

    outer_splits = make_strict_outer_splits(
        data,
        target_col=target_col,
        validation_rows=validation_rows,
        test_rows=test_rows,
        mature_min_rows=mature_min_rows,
        limited_history_min_block_rows=limited_history_min_block_rows,
        min_train_rows=min_train_rows,
        max_train_rows=max_train_rows,
    )
    for key, value in run_metadata.items():
        outer_splits[key] = value
    save_table(outer_splits, reports_dir / "outer_splits.parquet")

    validation_frames = []
    test_frames = []
    tuning_metric_frames = []
    best_param_rows = []
    split_lookup = outer_splits[outer_splits["status"] == "ok"].set_index("ticker").to_dict(orient="index")

    for ticker, split in split_lookup.items():
        ticker_data = data[data["ticker"].astype(str) == str(ticker)].sort_values("date").copy()
        train_raw = ticker_data[
            (ticker_data["date"] >= pd.Timestamp(split["train_start"]))
            & (ticker_data["date"] <= pd.Timestamp(split["train_end"]))
        ]
        train = _purge_train_target_overlap(train_raw, pd.Timestamp(split["validation_start"]))
        validation = ticker_data[
            (ticker_data["date"] >= pd.Timestamp(split["validation_start"]))
            & (ticker_data["date"] <= pd.Timestamp(split["validation_end"]))
        ]
        refit_raw = ticker_data[
            (ticker_data["date"] >= pd.Timestamp(split["refit_start"]))
            & (ticker_data["date"] <= pd.Timestamp(split["refit_end"]))
        ]
        refit_train = _purge_train_target_overlap(refit_raw, pd.Timestamp(split["test_start"]))
        test = ticker_data[
            (ticker_data["date"] >= pd.Timestamp(split["test_start"]))
            & (ticker_data["date"] <= pd.Timestamp(split["test_end"]))
        ]
        if train.empty or validation.empty or refit_train.empty or test.empty:
            continue

        for config in model_configs:
            model_name = config["name"]
            tuned_config = _limited_history_config(config, bool(split["limited_history"]), limited_history_n_trials)
            model_feature_cols = _model_feature_cols(tuned_config, feature_cols)
            model_hyperparams_dir = ensure_dir(hyperparams_dir / model_name)
            params, tuning_metrics = _get_or_search_inner_cv_params(
                tuned_config,
                train,
                model_feature_cols,
                target_col,
                model_hyperparams_dir,
                force_retrain=force_retrain,
                metric=primary_metric,
                random_state=random_state,
                ticker=str(ticker),
                run_metadata={
                    **run_metadata,
                    "ticker": ticker,
                    "training_scope": "per_ticker_strict",
                    "split_quality": split["split_quality"],
                    "limited_history": bool(split["limited_history"]),
                },
                target_col_name=target_col,
                execution_timing=execution_timing,
                inner_max_folds=inner_max_folds,
                inner_min_train_rows=min(inner_min_train_rows, max(10, len(train) // 2)),
                inner_validation_window=inner_validation_window or _default_inner_validation_window(horizon),
            )
            selected_params = _post_selection_params(tuned_config, params)
            best_param_rows.append(
                {
                    **run_metadata,
                    "ticker": ticker,
                    "model_name": model_name,
                    "best_params": selected_params,
                    "split_quality": split["split_quality"],
                    "limited_history": bool(split["limited_history"]),
                }
            )
            if not tuning_metrics.empty:
                tuning_metric_frames.append(tuning_metrics)

            validation_estimator = build_estimator(tuned_config, params=selected_params, random_state=random_state)
            _fit_model_estimator(validation_estimator, tuned_config, train, model_feature_cols, target_col)
            validation_pred = _strict_prediction_frame(
                validation,
                target_col=target_col,
                model_name=model_name,
                y_pred=_predict_model_estimator(validation_estimator, tuned_config, validation, model_feature_cols),
                split_role="validation",
                metadata={
                    **run_metadata,
                    "split_quality": split["split_quality"],
                    "limited_history": bool(split["limited_history"]),
                    "n_train": int(len(train)),
                    "n_train_before_purge": int(len(train_raw)),
                    "n_train_purged": int(len(train_raw) - len(train)),
                    "n_train_available": int(split["n_train_available"]),
                    "max_train_rows": split["max_train_rows"],
                    "train_start": train["date"].min(),
                    "train_end": train["date"].max(),
                    "validation_start": validation["date"].min(),
                    "validation_end": validation["date"].max(),
                },
            )
            validation_frames.append(validation_pred)

            final_path = models_dir / model_name / _safe_filename(str(ticker)) / "final.pkl"
            expected_final_metadata = _final_model_metadata(
                model_name=model_name,
                training_scope="per_ticker_strict_final",
                feature_cols=model_feature_cols,
                target_col=target_col,
                params=selected_params,
                train=refit_train,
                metadata={
                    **run_metadata,
                    "ticker": ticker,
                    "target_purge_policy": TARGET_PURGE_POLICY,
                    "refit_scope": "train_validation_purged_before_test",
                    "n_train_before_purge": int(len(refit_raw)),
                    "n_train_purged": int(len(refit_raw) - len(refit_train)),
                    "n_refit_available": int(split["n_refit_available"]),
                    "max_train_rows": split["max_train_rows"],
                    "train_window_start": refit_train["date"].min(),
                    "train_window_end": refit_train["date"].max(),
                    "validation_start": validation["date"].min(),
                    "validation_end": validation["date"].max(),
                    "test_start": test["date"].min(),
                    "test_end": test["date"].max(),
                    "max_train_target_date": refit_train["target_date"].max(),
                    "split_quality": split["split_quality"],
                    "limited_history": bool(split["limited_history"]),
                },
                ticker=str(ticker),
            )
            final_estimator = (
                _load_matching_model_estimator(final_path, expected_final_metadata)
                if not force_retrain and final_path.exists()
                else None
            )
            if final_estimator is None:
                final_estimator = build_estimator(tuned_config, params=selected_params, random_state=random_state)
                _fit_model_estimator(final_estimator, tuned_config, refit_train, model_feature_cols, target_col)
                _save_model_payload(final_path, {"estimator": final_estimator, **expected_final_metadata})

            test_pred = _strict_prediction_frame(
                test,
                target_col=target_col,
                model_name=model_name,
                y_pred=_predict_model_estimator(final_estimator, tuned_config, test, model_feature_cols),
                split_role="test",
                metadata={
                    **run_metadata,
                    "split_quality": split["split_quality"],
                    "limited_history": bool(split["limited_history"]),
                    "n_train": int(len(refit_train)),
                    "n_train_before_purge": int(len(refit_raw)),
                    "n_train_purged": int(len(refit_raw) - len(refit_train)),
                    "n_refit_available": int(split["n_refit_available"]),
                    "max_train_rows": split["max_train_rows"],
                    "train_start": refit_train["date"].min(),
                    "train_end": refit_train["date"].max(),
                    "test_start": test["date"].min(),
                    "test_end": test["date"].max(),
                    "max_train_target_date": refit_train["target_date"].max(),
                },
            )
            test_frames.append(test_pred)

    validation_predictions = pd.concat(validation_frames, ignore_index=True) if validation_frames else pd.DataFrame()
    test_predictions = pd.concat(test_frames, ignore_index=True) if test_frames else pd.DataFrame()
    tuning_metrics_df = pd.concat(tuning_metric_frames, ignore_index=True) if tuning_metric_frames else pd.DataFrame()
    tuning_metrics_df = _normalize_datetime_columns(
        tuning_metrics_df,
        ["inner_train_start", "inner_train_end", "inner_validation_start", "inner_validation_end"],
    )
    best_params_df = pd.DataFrame(best_param_rows)

    validation_metrics = _strict_metrics(validation_predictions, "validation")
    ranking = _validation_ranking(validation_metrics, primary_metric)
    selected_models = ranking[ranking["is_validation_selected"]].copy() if not ranking.empty else pd.DataFrame()
    test_metrics = _strict_metrics(test_predictions, "test")
    if not ranking.empty and not test_metrics.empty:
        enrich = ranking[["ticker", "model_name", "validation_rank", "is_validation_selected", f"validation_{primary_metric}"]]
        test_metrics = test_metrics.merge(enrich, on=["ticker", "model_name"], how="left")

    signal_equity, signal_metrics = _run_strict_signal_reports(
        validation_predictions,
        test_predictions,
        horizon=horizon,
        transaction_cost_bps=transaction_cost_bps,
        slippage_bps=slippage_bps,
        long_threshold=long_threshold,
        signal_anchor=signal_anchor,
    )
    leakage_audit = strict_protocol_leakage_audit(
        outer_splits=outer_splits,
        validation_predictions=validation_predictions,
        test_predictions=test_predictions,
        models_dir=models_dir,
    )

    save_table(validation_predictions, reports_dir / "validation_predictions.parquet")
    save_table(test_predictions, reports_dir / "test_predictions.parquet")
    save_table(validation_metrics, reports_dir / "validation_prediction_metrics.parquet")
    save_table(ranking, reports_dir / "validation_model_ranking.parquet")
    save_table(selected_models, reports_dir / "selected_models_by_ticker.parquet")
    save_table(test_metrics, reports_dir / "test_prediction_metrics.parquet")
    save_table(signal_equity, reports_dir / "test_signal_equity.parquet")
    save_table(signal_metrics, reports_dir / "test_signal_metrics.parquet")
    save_table(leakage_audit, reports_dir / "leakage_audit.parquet")
    if not tuning_metrics_df.empty:
        save_table(tuning_metrics_df, reports_dir / "inner_tuning_metrics.parquet")
    if not best_params_df.empty:
        best_params_table = best_params_df.copy()
        best_params_table["best_params"] = best_params_table["best_params"].map(lambda value: json.dumps(value, sort_keys=True))
        save_table(best_params_table, reports_dir / "best_params.parquet")
        save_json(best_params_df.to_dict(orient="records"), reports_dir / "best_params.json")

    return {
        "artifact_root": artifact_root,
        "reports_dir": reports_dir,
        "models_dir": models_dir,
        "outer_splits": outer_splits,
        "validation_predictions": validation_predictions,
        "test_predictions": test_predictions,
        "validation_prediction_metrics": validation_metrics,
        "validation_model_ranking": ranking,
        "selected_models_by_ticker": selected_models,
        "test_prediction_metrics": test_metrics,
        "test_signal_equity": signal_equity,
        "test_signal_metrics": signal_metrics,
        "inner_tuning_metrics": tuning_metrics_df,
        "best_params": best_params_df,
        "leakage_audit": leakage_audit,
    }


def strict_protocol_leakage_audit(
    outer_splits: pd.DataFrame,
    validation_predictions: pd.DataFrame,
    test_predictions: pd.DataFrame,
    models_dir: Path,
) -> pd.DataFrame:
    rows = []
    rows.append(_audit_row("strict outer splits are available", not outer_splits.empty, f"split_rows={len(outer_splits)}"))
    rows.append(_audit_row("validation predictions are available", not validation_predictions.empty, f"rows={len(validation_predictions)}"))
    rows.append(_audit_row("test predictions are available", not test_predictions.empty, f"rows={len(test_predictions)}"))

    if not outer_splits.empty:
        ok_splits = outer_splits[outer_splits["status"] == "ok"].copy()
        order_ok = (
            pd.to_datetime(ok_splits["train_end"]) < pd.to_datetime(ok_splits["validation_start"])
        ) & (
            pd.to_datetime(ok_splits["validation_end"]) < pd.to_datetime(ok_splits["test_start"])
        )
        rows.append(_audit_row("outer split dates are chronological", bool(order_ok.all()), f"bad_rows={int((~order_ok).sum())}"))
        if "max_train_rows" in ok_splits.columns:
            capped = ok_splits["max_train_rows"].notna()
            train_over_cap = int((ok_splits.loc[capped, "n_train"] > ok_splits.loc[capped, "max_train_rows"]).sum())
            refit_over_cap = int((ok_splits.loc[capped, "n_refit"] > ok_splits.loc[capped, "max_train_rows"]).sum())
            rows.append(_audit_row("train and refit windows respect max_train_rows", train_over_cap == 0 and refit_over_cap == 0, f"train_over_cap={train_over_cap}, refit_over_cap={refit_over_cap}"))

    for split_role, predictions, start_col, end_col in [
        ("validation", validation_predictions, "validation_start", "validation_end"),
        ("test", test_predictions, "test_start", "test_end"),
    ]:
        if predictions.empty or outer_splits.empty:
            continue
        bounds = outer_splits[["ticker", start_col, end_col]].rename(
            columns={start_col: "_audit_start", end_col: "_audit_end"}
        )
        merged = predictions.merge(bounds, on="ticker", how="left")
        dates = pd.to_datetime(merged["date"])
        start = pd.to_datetime(merged["_audit_start"])
        end = pd.to_datetime(merged["_audit_end"])
        out_of_window = int(((dates < start) | (dates > end)).sum())
        wrong_role = int((merged["split_role"] != split_role).sum()) if "split_role" in merged.columns else len(merged)
        rows.append(_audit_row(f"{split_role} predictions match outer split dates", out_of_window == 0 and wrong_role == 0, f"out_of_window={out_of_window}, wrong_role={wrong_role}"))

    if not test_predictions.empty and "max_train_target_date" in test_predictions.columns:
        overlap = int((pd.to_datetime(test_predictions["max_train_target_date"]) >= pd.to_datetime(test_predictions["test_start"])).sum())
        rows.append(_audit_row("final refit target dates end before test starts", overlap == 0, f"overlap_rows={overlap}"))

    expected_models = test_predictions[["model_name", "ticker"]].drop_duplicates() if not test_predictions.empty else pd.DataFrame()
    missing_models = 0
    for row in expected_models.itertuples(index=False):
        path = models_dir / str(row.model_name) / _safe_filename(str(row.ticker)) / "final.pkl"
        if not path.exists():
            missing_models += 1
    rows.append(_audit_row("final model payloads exist for test predictions", missing_models == 0, f"missing_models={missing_models}"))
    return pd.DataFrame(rows)


def _get_or_search_inner_cv_params(
    model_config: dict[str, Any],
    train: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    cache_dir: Path,
    force_retrain: bool,
    metric: str,
    random_state: int,
    ticker: str,
    run_metadata: dict[str, Any],
    target_col_name: str,
    execution_timing: str,
    inner_max_folds: int,
    inner_min_train_rows: int,
    inner_validation_window: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    cache_path = cache_dir / f"{_safe_filename(ticker)}.json"
    cache_metadata = {
        **run_metadata,
        **_feature_set_metadata(feature_cols),
        "target_col": target_col_name,
        "execution_timing": execution_timing,
        "inner_max_folds": int(inner_max_folds),
        "inner_min_train_rows": int(inner_min_train_rows),
        "inner_validation_window": int(inner_validation_window),
        "train_start": train["date"].min(),
        "train_end": train["date"].max(),
        "n_train": int(len(train)),
    }
    search_space = normalize_search_space(model_config)
    static_params = dict(model_config.get("static_params", {}))
    n_trials = int(model_config.get("n_trials", 25))
    optuna_n_jobs = int(model_config.get("optuna_n_jobs", 1))
    tuning_backend = "optuna_inner_cv" if search_space else "fixed"

    if cache_path.exists() and not force_retrain:
        cached = load_json(cache_path)
        if _strict_hyperparameter_cache_matches(
            cached,
            model_config,
            metric,
            search_space,
            n_trials,
            tuning_backend,
            cache_metadata,
        ):
            return dict(cached["best_params"]), pd.DataFrame(cached.get("inner_fold_metrics", []))

    inner_splits = _inner_time_series_splits(
        train,
        validation_window=inner_validation_window,
        max_folds=inner_max_folds,
        min_train_rows=inner_min_train_rows,
    )
    if not search_space or not inner_splits:
        best_params = static_params
        fold_metrics = _score_inner_folds(model_config, best_params, train, feature_cols, target_col, inner_splits, random_state, metric, ticker)
        _save_strict_hyperparameter_payload(
            cache_path,
            model_config,
            metric,
            static_params,
            search_space,
            n_trials,
            optuna_n_jobs,
            "fixed" if not search_space else "static_fallback",
            best_params,
            fold_metrics,
            cache_metadata,
        )
        return best_params, fold_metrics

    try:
        import optuna
    except ImportError as exc:
        raise ImportError("Optuna is required for strict inner-CV hyperparameter search.") from exc
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    direction = "minimize" if METRIC_DIRECTIONS.get(metric, "min") == "min" else "maximize"

    def objective(trial: Any) -> float:
        params = static_params | suggest_params(trial, search_space)
        fold_metrics = _score_inner_folds(model_config, params, train, feature_cols, target_col, inner_splits, random_state, metric, ticker)
        score = float(fold_metrics[metric].mean()) if not fold_metrics.empty else float("nan")
        for key in ["mae", "rmse", "r2", "pearson", "spearman", "directional_accuracy"]:
            if key in fold_metrics.columns:
                trial.set_user_attr(key, float(fold_metrics[key].mean()))
        if np.isfinite(score):
            return score
        return float("inf") if direction == "minimize" else float("-inf")

    study = optuna.create_study(direction=direction, sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials, n_jobs=optuna_n_jobs)
    best_params = static_params | dict(study.best_trial.params)
    fold_metrics = _score_inner_folds(model_config, best_params, train, feature_cols, target_col, inner_splits, random_state, metric, ticker)
    _save_strict_hyperparameter_payload(
        cache_path,
        model_config,
        metric,
        static_params,
        search_space,
        n_trials,
        optuna_n_jobs,
        tuning_backend,
        best_params,
        fold_metrics,
        {
            **cache_metadata,
            "best_score": float(study.best_value),
            "best_trial_number": int(study.best_trial.number),
        },
    )
    return best_params, fold_metrics


def _score_inner_folds(
    model_config: dict[str, Any],
    params: dict[str, Any],
    train: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    inner_splits: list[dict[str, int]],
    random_state: int,
    metric: str,
    ticker: str,
) -> pd.DataFrame:
    rows = []
    for split in inner_splits:
        inner_train_raw = train.iloc[: split["train_end"]].copy()
        inner_valid = train.iloc[split["valid_start"] : split["valid_end"]].copy()
        if inner_valid.empty:
            continue
        inner_train = _purge_train_target_overlap(inner_train_raw, inner_valid["date"].min())
        if inner_train.empty:
            continue
        estimator = build_estimator(model_config, params=params, random_state=random_state)
        _fit_model_estimator(estimator, model_config, inner_train, feature_cols, target_col)
        pred = _predict_model_estimator(estimator, model_config, inner_valid, feature_cols)
        metrics = regression_metrics(inner_valid[target_col], pred)
        rows.append(
            {
                "ticker": ticker,
                "model_name": model_config["name"],
                "inner_fold": int(split["inner_fold"]),
                "inner_train_start": inner_train["date"].min(),
                "inner_train_end": inner_train["date"].max(),
                "inner_validation_start": inner_valid["date"].min(),
                "inner_validation_end": inner_valid["date"].max(),
                "n_inner_train": int(len(inner_train)),
                "n_inner_train_before_purge": int(len(inner_train_raw)),
                "n_inner_train_purged": int(len(inner_train_raw) - len(inner_train)),
                "n_inner_valid": int(len(inner_valid)),
                **metrics,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty and metric:
        return out
    return out


def _inner_time_series_splits(
    train: pd.DataFrame,
    validation_window: int,
    max_folds: int,
    min_train_rows: int,
) -> list[dict[str, int]]:
    n_obs = len(train)
    if n_obs <= min_train_rows + 1:
        return []
    validation_window = min(validation_window, max(1, (n_obs - min_train_rows) // 2 or 1))
    splits = []
    valid_end = n_obs
    while valid_end - validation_window >= min_train_rows and len(splits) < max_folds:
        valid_start = valid_end - validation_window
        splits.append(
            {
                "inner_fold": len(splits),
                "train_end": int(valid_start),
                "valid_start": int(valid_start),
                "valid_end": int(valid_end),
            }
        )
        valid_end = valid_start
    return list(reversed(splits))


def _model_feature_cols(model_config: dict[str, Any], default_feature_cols: list[str]) -> list[str]:
    return list(model_config.get("feature_cols") or default_feature_cols)


def _model_input_mode(model_config: dict[str, Any]) -> str:
    return str(model_config.get("input_mode", "features"))


def _model_input_frame(data: pd.DataFrame, feature_cols: list[str], model_config: dict[str, Any]) -> pd.DataFrame:
    if _model_input_mode(model_config) == "full_frame":
        metadata_cols = [col for col in ["date", "ticker", "target_date", "entry_date"] if col in data.columns]
        cols = [*metadata_cols, *feature_cols]
        return data.loc[:, cols].copy()
    return data.loc[:, feature_cols]


def _fit_model_estimator(
    estimator: Any,
    model_config: dict[str, Any],
    train: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
) -> None:
    estimator.fit(_model_input_frame(train, feature_cols, model_config), train[target_col])


def _predict_model_estimator(
    estimator: Any,
    model_config: dict[str, Any],
    data: pd.DataFrame,
    feature_cols: list[str],
) -> np.ndarray:
    return estimator.predict(_model_input_frame(data, feature_cols, model_config))


def _strict_prediction_frame(
    data: pd.DataFrame,
    target_col: str,
    model_name: str,
    y_pred: np.ndarray,
    split_role: str,
    metadata: dict[str, Any] | None = None,
) -> pd.DataFrame:
    optional_cols = ["close", "future_close", "entry_date", "entry_open", "future_open", "target_date"]
    pred_cols = ["date", "ticker", *[col for col in optional_cols if col in data.columns], target_col]
    pred = data[pred_cols].copy()
    pred["model_name"] = model_name
    pred["split_role"] = split_role
    pred["y_true"] = data[target_col].to_numpy()
    pred["y_pred"] = y_pred
    for key, value in dict(metadata or {}).items():
        pred[key] = value
    return pred


def _strict_metrics(predictions: pd.DataFrame, split_role: str) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    metrics = grouped_regression_metrics(predictions, ["ticker", "model_name"])
    metrics.insert(0, "split_role", split_role)
    enrich_cols = [
        "ticker",
        "model_name",
        "horizon_name",
        "horizon",
        "split_quality",
        "limited_history",
        "n_train",
        "n_train_before_purge",
        "n_train_purged",
        "n_train_available",
        "n_refit_available",
        "max_train_rows",
        "train_start",
        "train_end",
    ]
    available = [col for col in enrich_cols if col in predictions.columns]
    if available:
        enrich = predictions[available].drop_duplicates(["ticker", "model_name"])
        metrics = metrics.merge(enrich, on=["ticker", "model_name"], how="left")
    return metrics


def _normalize_datetime_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    for column in columns:
        if column in out.columns:
            out[column] = pd.to_datetime(out[column], errors="coerce")
    return out


def _validation_ranking(validation_metrics: pd.DataFrame, primary_metric: str) -> pd.DataFrame:
    if validation_metrics.empty:
        return pd.DataFrame()
    ascending = METRIC_DIRECTIONS.get(primary_metric, "min") == "min"
    ranking = validation_metrics.copy()
    ranking = ranking.rename(columns={primary_metric: f"validation_{primary_metric}"})
    sort_cols = ["ticker", f"validation_{primary_metric}"]
    sort_ascending = [True, ascending]
    if primary_metric != "directional_accuracy" and "directional_accuracy" in ranking.columns:
        sort_cols.append("directional_accuracy")
        sort_ascending.append(False)
    if primary_metric != "rmse" and "rmse" in ranking.columns:
        sort_cols.append("rmse")
        sort_ascending.append(True)
    sort_cols.append("model_name")
    sort_ascending.append(True)
    ranking = ranking.sort_values(sort_cols, ascending=sort_ascending)
    ranking["validation_rank"] = ranking.groupby("ticker").cumcount() + 1
    ranking["is_validation_selected"] = ranking["validation_rank"].eq(1)
    return ranking


def _run_strict_signal_reports(
    validation_predictions: pd.DataFrame,
    test_predictions: pd.DataFrame,
    horizon: int,
    transaction_cost_bps: float,
    slippage_bps: float,
    long_threshold: float,
    signal_anchor: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    equity_frames = []
    metric_frames = []
    for mode in ["overlapping_tranches", "non_overlapping"]:
        equity, metrics = run_test_signal_backtest(
            validation_predictions,
            test_predictions,
            horizon=horizon,
            mode=mode,
            transaction_cost_bps=transaction_cost_bps,
            slippage_bps=slippage_bps,
            long_threshold=long_threshold,
            signal_anchor=signal_anchor,
        )
        if not equity.empty:
            equity_frames.append(equity)
        if not metrics.empty:
            metric_frames.append(metrics)
    equity_df = pd.concat(equity_frames, ignore_index=True) if equity_frames else pd.DataFrame()
    metrics_df = pd.concat(metric_frames, ignore_index=True) if metric_frames else pd.DataFrame()
    return equity_df, metrics_df


def _save_strict_hyperparameter_payload(
    cache_path: Path,
    model_config: dict[str, Any],
    metric: str,
    static_params: dict[str, Any],
    search_space: dict[str, dict[str, Any]],
    n_trials: int,
    optuna_n_jobs: int,
    tuning_backend: str,
    best_params: dict[str, Any],
    fold_metrics: pd.DataFrame,
    cache_metadata: dict[str, Any],
) -> None:
    payload = {
        "model_name": model_config["name"],
        "metric": metric,
        "static_params": static_params,
        "search_space": search_space,
        "n_trials": int(n_trials),
        "optuna_n_jobs": int(optuna_n_jobs),
        "tuning_backend": tuning_backend,
        "target_purge_policy": TARGET_PURGE_POLICY,
        "best_params": best_params,
        "inner_fold_metrics": fold_metrics.to_dict(orient="records"),
    }
    payload.update(cache_metadata)
    save_json(payload, cache_path)


def _strict_hyperparameter_cache_matches(
    cached: dict[str, Any],
    model_config: dict[str, Any],
    metric: str,
    search_space: dict[str, dict[str, Any]],
    n_trials: int,
    tuning_backend: str,
    cache_metadata: dict[str, Any],
) -> bool:
    expected_backends = {tuning_backend}
    if tuning_backend == "optuna_inner_cv":
        expected_backends.add("static_fallback")
    if cached.get("model_name") != model_config["name"]:
        return False
    if cached.get("metric") != metric:
        return False
    if cached.get("static_params") != dict(model_config.get("static_params", {})):
        return False
    if cached.get("search_space") != search_space:
        return False
    if cached.get("n_trials") != int(n_trials):
        return False
    if cached.get("tuning_backend") not in expected_backends:
        return False
    for key, value in cache_metadata.items():
        if not _metadata_values_match(cached.get(key), value):
            return False
    return True


def _limited_history_config(config: dict[str, Any], limited_history: bool, limited_history_n_trials: int) -> dict[str, Any]:
    out = dict(config)
    if limited_history and "limited_history_search_space" in config:
        out["search_space"] = dict(config["limited_history_search_space"])
    elif not limited_history and "mature_history_search_space" in config:
        out["search_space"] = dict(config["mature_history_search_space"])
    if limited_history and "limited_history_static_params" in config:
        out["static_params"] = dict(config.get("static_params", {})) | dict(config["limited_history_static_params"])
    elif not limited_history and "mature_history_static_params" in config:
        out["static_params"] = dict(config.get("static_params", {})) | dict(config["mature_history_static_params"])
    if limited_history and "n_trials" in config:
        model_limit = int(config.get("limited_history_n_trials", limited_history_n_trials))
        out["n_trials"] = min(int(config.get("n_trials", model_limit)), model_limit)
    return out


def _post_selection_params(config: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    return dict(params) | dict(config.get("post_selection_static_params", {}))


def _default_inner_validation_window(horizon: int) -> int:
    return 63 if horizon >= 21 else 42


def _infer_horizon(target_col: str) -> int:
    parts = target_col.split("_")
    return int(parts[2])


def _safe_filename(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value)).strip("._") or "ticker"


def _audit_row(check: str, passed: bool, details: str) -> dict[str, Any]:
    return {"check": check, "passed": bool(passed), "details": details}


def _metadata_values_match(left: Any, right: Any) -> bool:
    if _is_datetime_like(left) or _is_datetime_like(right):
        return pd.Timestamp(left) == pd.Timestamp(right)
    if hasattr(left, "item"):
        left = left.item()
    if hasattr(right, "item"):
        right = right.item()
    return left == right


def _is_datetime_like(value: Any) -> bool:
    if isinstance(value, pd.Timestamp):
        return True
    if isinstance(value, str):
        try:
            pd.Timestamp(value)
            return True
        except ValueError:
            return False
    return hasattr(value, "to_datetime64")
