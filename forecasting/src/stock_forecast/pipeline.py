from __future__ import annotations

import hashlib
from itertools import product
from pathlib import Path
import re
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd

from .artifacts import load_json, save_json, save_table
from .metrics import grouped_regression_metrics, regression_metrics
from .models import ModelSpec, build_model, make_pipeline
from .utils import ensure_dir

METRIC_DIRECTIONS = {
    "mae": "min",
    "rmse": "min",
    "r2": "max",
    "pearson": "max",
    "spearman": "max",
    "directional_accuracy": "max",
}

TARGET_PURGE_POLICY = "target_date_before_validation_start"
STATIC_FALLBACK_TUNING_BACKEND = "static_fallback"


def get_or_search_best_params(
    model_config: dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    cache_dir: str | Path,
    force_retrain: bool = False,
    metric: str = "rmse",
    validation_fraction: float = 0.2,
    random_state: int = 42,
    feature_dates: pd.Series | None = None,
    target_dates: pd.Series | None = None,
    target_col: str | None = None,
    execution_timing: str | None = None,
    cache_key: str | None = None,
    cache_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cache_path = Path(cache_dir) / f"{cache_key or model_config['name']}.json"
    cache_metadata = {
        **dict(cache_metadata or {}),
        **_feature_set_metadata(list(X.columns)),
    }
    search_space = normalize_search_space(model_config)
    n_trials = int(model_config.get("n_trials", 25))
    optuna_n_jobs = int(model_config.get("optuna_n_jobs", 1))
    tuning_backend = "optuna" if search_space else "fixed"
    if cache_path.exists() and not force_retrain:
        cached = load_json(cache_path)
        if "best_params" in cached and _hyperparameter_cache_matches(
            cached,
            model_config,
            metric,
            target_dates,
            target_col,
            execution_timing,
            search_space,
            tuning_backend,
            n_trials,
            cache_metadata=cache_metadata,
        ):
            return dict(cached["best_params"])

    static_params = dict(model_config.get("static_params", {}))
    if not search_space:
        best_params = static_params
        save_json(
            _hyperparameter_payload(
                model_config["name"],
                metric,
                static_params,
                dict(model_config.get("search_params", {})),
                search_space,
                n_trials,
                optuna_n_jobs,
                tuning_backend,
                model_config.get("max_trials"),
                TARGET_PURGE_POLICY if target_dates is not None else None,
                target_col,
                execution_timing,
                best_params,
                [],
                [],
                best_trial_number=None,
                cache_metadata=cache_metadata,
            ),
            cache_path,
        )
        return best_params

    try:
        X_inner_train, y_inner_train, X_inner_valid, y_inner_valid, inner_validation_start = chronological_holdout(
            X, y, feature_dates=feature_dates, validation_fraction=validation_fraction
        )
    except ValueError as exc:
        if str(exc) != "Not enough rows for chronological holdout":
            raise
        return _save_static_hyperparameter_fallback(
            model_config,
            metric,
            static_params,
            search_space,
            n_trials,
            optuna_n_jobs,
            cache_path,
            reason=str(exc),
            target_dates=target_dates,
            target_col=target_col,
            execution_timing=execution_timing,
            cache_metadata=cache_metadata,
        )
    n_inner_train_before_purge = len(X_inner_train)
    if target_dates is not None:
        inner_target_dates = pd.Series(target_dates).reindex(X_inner_train.index)
        keep = inner_target_dates < inner_validation_start
        X_inner_train = X_inner_train.loc[keep]
        y_inner_train = y_inner_train.loc[keep]
        if X_inner_train.empty:
            return _save_static_hyperparameter_fallback(
                model_config,
                metric,
                static_params,
                search_space,
                n_trials,
                optuna_n_jobs,
                cache_path,
                reason="purged_hyperparameter_train_split_empty",
                target_dates=target_dates,
                target_col=target_col,
                execution_timing=execution_timing,
                cache_metadata=cache_metadata,
                inner_validation_start=inner_validation_start,
                n_inner_train_before_purge=n_inner_train_before_purge,
                n_inner_train_after_purge=len(X_inner_train),
                n_inner_valid=len(X_inner_valid),
            )

    direction = "minimize" if METRIC_DIRECTIONS.get(metric, "min") == "min" else "maximize"

    try:
        import optuna
    except ImportError as exc:
        raise ImportError("Optuna is required for hyperparameter search. Install the forecasting package dependencies.") from exc
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: Any) -> float:
        params = static_params | suggest_params(trial, search_space)
        estimator = build_estimator(model_config, params=params, random_state=random_state)
        estimator.fit(X_inner_train, y_inner_train)
        pred = estimator.predict(X_inner_valid)
        scores = regression_metrics(y_inner_valid, pred)
        for key, value in scores.items():
            trial.set_user_attr(key, value)
        score = scores[metric]
        if np.isfinite(score):
            return float(score)
        return float("inf") if direction == "minimize" else float("-inf")

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction=direction, sampler=sampler)
    study.optimize(objective, n_trials=n_trials, n_jobs=optuna_n_jobs)

    completed_trials = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    if not completed_trials:
        raise RuntimeError(f"Optuna search produced no completed trials for model {model_config['name']}")

    best_trial = study.best_trial
    best_params = static_params | dict(best_trial.params)
    best_score = float(best_trial.value)
    trials = optuna_trials_payload(study.trials)
    search_results = [
        {"params": trial["params"], **trial["metrics"]}
        for trial in trials
        if trial["state"] == "COMPLETE" and trial["value"] is not None
    ]

    save_json(
        _hyperparameter_payload(
            model_config["name"],
            metric,
            static_params,
            dict(model_config.get("search_params", {})),
            search_space,
            n_trials,
            optuna_n_jobs,
            tuning_backend,
            model_config.get("max_trials"),
            TARGET_PURGE_POLICY if target_dates is not None else None,
            target_col,
            execution_timing,
            best_params,
            search_results,
            trials,
            best_score=best_score,
            best_trial_number=int(best_trial.number),
            cache_metadata=cache_metadata,
        ),
        cache_path,
    )
    return best_params


def run_walk_forward_training(
    model_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    splits: list[dict[str, pd.DatetimeIndex]],
    model_configs: list[dict[str, Any]],
    artifact_dir: str | Path,
    force_retrain: bool = False,
    primary_metric: str = "directional_accuracy",
    random_state: int = 42,
    train_by_ticker: bool = False,
    run_metadata: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    artifact_root = Path(artifact_dir)
    hyperparams_dir = ensure_dir(artifact_root / "hyperparams")
    models_dir = ensure_dir(artifact_root / "models")
    predictions_dir = ensure_dir(artifact_root / "predictions")
    reports_dir = ensure_dir(artifact_root / "reports")
    run_metadata = dict(run_metadata or {})

    if not splits:
        raise ValueError("At least one walk-forward split is required")

    model_df = _ensure_target_date(model_df, target_col)
    execution_timing = _target_execution_timing(target_col)
    tickers = sorted(model_df["ticker"].dropna().astype(str).unique())
    first_train_raw = model_df[model_df["date"].isin(splits[0]["train_dates"])].sort_values(["date", "ticker"])
    first_train = _purge_train_target_overlap(first_train_raw, splits[0]["validation_dates"].min())
    best_params_by_model: dict[str, Any] = {}
    for config in model_configs:
        model_name = config["name"]
        if train_by_ticker:
            best_params_by_model[model_name] = {}
            model_hyperparams_dir = ensure_dir(hyperparams_dir / model_name)
            for ticker in tickers:
                ticker_first_train = _first_available_ticker_train(model_df, splits, ticker)
                if ticker_first_train.empty:
                    continue
                best_params_by_model[model_name][ticker] = get_or_search_best_params(
                    config,
                    ticker_first_train[feature_cols],
                    ticker_first_train[target_col],
                    model_hyperparams_dir,
                    force_retrain=force_retrain,
                    metric=primary_metric,
                    random_state=random_state,
                    feature_dates=ticker_first_train["date"],
                    target_dates=ticker_first_train["target_date"],
                    target_col=target_col,
                    execution_timing=execution_timing,
                    cache_key=safe_filename(ticker),
                    cache_metadata={**run_metadata, "training_scope": "per_ticker", "ticker": ticker},
                )
        else:
            best_params_by_model[model_name] = get_or_search_best_params(
                config,
                first_train[feature_cols],
                first_train[target_col],
                hyperparams_dir,
                force_retrain=force_retrain,
                metric=primary_metric,
                random_state=random_state,
                feature_dates=first_train["date"],
                target_dates=first_train["target_date"],
                target_col=target_col,
                execution_timing=execution_timing,
                cache_metadata={**run_metadata, "training_scope": "global"},
            )

    all_predictions = []
    all_metrics = []
    all_ticker_metrics = []
    for config in model_configs:
        model_name = config["name"]
        model_predictions = []
        model_metrics = []
        model_ticker_metrics = []
        for fold_idx, split in enumerate(splits):
            train_raw = model_df[model_df["date"].isin(split["train_dates"])].sort_values(["date", "ticker"])
            train = _purge_train_target_overlap(train_raw, split["validation_dates"].min())
            valid = model_df[model_df["date"].isin(split["validation_dates"])].sort_values(["date", "ticker"])
            if train.empty or valid.empty:
                continue

            if train_by_ticker:
                fold_predictions = []
                for ticker in tickers:
                    ticker_train_raw = train_raw[train_raw["ticker"].astype(str) == ticker]
                    ticker_train = train[train["ticker"].astype(str) == ticker]
                    ticker_valid = valid[valid["ticker"].astype(str) == ticker]
                    params = best_params_by_model[model_name].get(ticker)
                    if ticker_train.empty or ticker_valid.empty or params is None:
                        continue

                    fold_path = models_dir / model_name / safe_filename(ticker) / f"fold_{fold_idx}.pkl"
                    expected_metadata = _fold_model_metadata(
                        model_name=model_name,
                        fold_idx=fold_idx,
                        training_scope="per_ticker",
                        feature_cols=feature_cols,
                        target_col=target_col,
                        params=params,
                        train=ticker_train,
                        train_raw=ticker_train_raw,
                        valid=ticker_valid,
                        metadata=run_metadata,
                        ticker=ticker,
                    )
                    estimator = (
                        _load_matching_model_estimator(fold_path, expected_metadata)
                        if not force_retrain and fold_path.exists()
                        else None
                    )
                    if estimator is None:
                        estimator = build_estimator(config, params=params, random_state=random_state)
                        estimator.fit(ticker_train[feature_cols], ticker_train[target_col])
                        _save_model_payload(
                            fold_path,
                            {
                                "estimator": estimator,
                                **expected_metadata,
                            },
                        )
                    y_pred = estimator.predict(ticker_valid[feature_cols])
                    pred = _prediction_frame(ticker_valid, target_col, fold_idx, model_name, y_pred, run_metadata)
                    model_predictions.append(pred)
                    all_predictions.append(pred)
                    fold_predictions.append(pred)

                    metrics = _fold_metrics(
                        pred,
                        split,
                        model_name,
                        len(ticker_train),
                        len(ticker_train_raw),
                        ticker=ticker,
                    )
                    metrics.update(run_metadata)
                    model_metrics.append(metrics)
                    all_metrics.append(metrics)

                    ticker_metrics = grouped_regression_metrics(pred, ["model_name", "fold", "ticker"])
                    ticker_metrics["validation_start"] = split["validation_dates"].min()
                    ticker_metrics["validation_end"] = split["validation_dates"].max()
                    for key, value in run_metadata.items():
                        ticker_metrics[key] = value
                    model_ticker_metrics.append(ticker_metrics)
                    all_ticker_metrics.append(ticker_metrics)

                if fold_predictions:
                    pooled_pred = pd.concat(fold_predictions, ignore_index=True)
                    pooled_metrics = _fold_metrics(
                        pooled_pred,
                        split,
                        model_name,
                        len(train),
                        len(train_raw),
                    )
                    pooled_metrics["training_scope"] = "per_ticker_aggregate"
                    pooled_metrics.update(run_metadata)
                    model_metrics.append(pooled_metrics)
                    all_metrics.append(pooled_metrics)
            else:
                fold_path = models_dir / model_name / f"fold_{fold_idx}.pkl"
                expected_metadata = _fold_model_metadata(
                    model_name=model_name,
                    fold_idx=fold_idx,
                    training_scope="global",
                    feature_cols=feature_cols,
                    target_col=target_col,
                    params=best_params_by_model[model_name],
                    train=train,
                    train_raw=train_raw,
                    valid=valid,
                    metadata=run_metadata,
                )
                estimator = (
                    _load_matching_model_estimator(fold_path, expected_metadata)
                    if not force_retrain and fold_path.exists()
                    else None
                )
                if estimator is None:
                    estimator = build_estimator(config, params=best_params_by_model[model_name], random_state=random_state)
                    estimator.fit(train[feature_cols], train[target_col])
                    _save_model_payload(
                        fold_path,
                        {
                            "estimator": estimator,
                            **expected_metadata,
                        },
                    )
                y_pred = estimator.predict(valid[feature_cols])

                pred = _prediction_frame(valid, target_col, fold_idx, model_name, y_pred, run_metadata)
                model_predictions.append(pred)
                all_predictions.append(pred)

                metrics = _fold_metrics(pred, split, model_name, len(train), len(train_raw))
                metrics.update(run_metadata)
                model_metrics.append(metrics)
                all_metrics.append(metrics)

                ticker_metrics = grouped_regression_metrics(pred, ["model_name", "fold", "ticker"])
                ticker_metrics["validation_start"] = split["validation_dates"].min()
                ticker_metrics["validation_end"] = split["validation_dates"].max()
                for key, value in run_metadata.items():
                    ticker_metrics[key] = value
                model_ticker_metrics.append(ticker_metrics)
                all_ticker_metrics.append(ticker_metrics)

        if train_by_ticker:
            for ticker in tickers:
                params = best_params_by_model[model_name].get(ticker)
                ticker_train = model_df[model_df["ticker"].astype(str) == ticker].dropna(subset=[target_col])
                if ticker_train.empty or params is None:
                    continue

                final_path = models_dir / model_name / safe_filename(ticker) / "final.pkl"
                expected_final_metadata = _final_model_metadata(
                    model_name=model_name,
                    training_scope="per_ticker",
                    feature_cols=feature_cols,
                    target_col=target_col,
                    params=params,
                    train=ticker_train,
                    metadata=run_metadata,
                    ticker=ticker,
                )
                if (
                    final_path.exists()
                    and not force_retrain
                    and _model_payload_matches(final_path, expected_final_metadata)
                ):
                    continue

                final_estimator = build_estimator(config, params=params, random_state=random_state)
                final_estimator.fit(ticker_train[feature_cols], ticker_train[target_col])
                _save_model_payload(
                    final_path,
                    {
                        "estimator": final_estimator,
                        **expected_final_metadata,
                    },
                )
        else:
            final_train = model_df.dropna(subset=[target_col])
            if not final_train.empty:
                final_path = models_dir / model_name / "final.pkl"
                expected_final_metadata = _final_model_metadata(
                    model_name=model_name,
                    training_scope="global",
                    feature_cols=feature_cols,
                    target_col=target_col,
                    params=best_params_by_model[model_name],
                    train=final_train,
                    metadata=run_metadata,
                )
                if (
                    force_retrain
                    or not final_path.exists()
                    or not _model_payload_matches(final_path, expected_final_metadata)
                ):
                    final_estimator = build_estimator(
                        config,
                        params=best_params_by_model[model_name],
                        random_state=random_state,
                    )
                    final_estimator.fit(final_train[feature_cols], final_train[target_col])
                    _save_model_payload(
                        final_path,
                        {
                            "estimator": final_estimator,
                            **expected_final_metadata,
                        },
                    )

        if model_predictions:
            model_pred_df = pd.concat(model_predictions, ignore_index=True)
            model_metrics_df = pd.DataFrame(model_metrics)
            save_table(model_pred_df, predictions_dir / f"{model_name}_oof.parquet")
            save_json(_metrics_payload(model_metrics_df), reports_dir / f"{model_name}_metrics.json")
            if model_ticker_metrics:
                model_ticker_metrics_df = pd.concat(model_ticker_metrics, ignore_index=True)
                save_table(model_ticker_metrics_df, reports_dir / f"{model_name}_ticker_metrics.parquet")

    predictions_df = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()
    metrics_df = pd.DataFrame(all_metrics)
    ticker_metrics_df = pd.concat(all_ticker_metrics, ignore_index=True) if all_ticker_metrics else pd.DataFrame()
    save_table(predictions_df, predictions_dir / "all_oof_predictions.parquet")
    save_table(metrics_df, reports_dir / "all_model_metrics.parquet")
    save_table(ticker_metrics_df, reports_dir / "all_model_ticker_metrics.parquet")
    return predictions_df, metrics_df, best_params_by_model


def _load_matching_model_estimator(path: Path, expected_metadata: dict[str, Any]):
    payload = joblib.load(path)
    if (
        isinstance(payload, dict)
        and "estimator" in payload
        and _model_metadata_matches(payload, expected_metadata)
    ):
        return payload["estimator"]
    return None


def _model_payload_matches(path: Path, expected_metadata: dict[str, Any]) -> bool:
    payload = joblib.load(path)
    return isinstance(payload, dict) and _model_metadata_matches(payload, expected_metadata)


def _save_model_payload(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    joblib.dump(payload, path)


def _fold_model_metadata(
    model_name: str,
    fold_idx: int,
    training_scope: str,
    feature_cols: list[str],
    target_col: str,
    params: dict[str, Any],
    train: pd.DataFrame,
    train_raw: pd.DataFrame,
    valid: pd.DataFrame,
    metadata: dict[str, Any] | None = None,
    ticker: str | None = None,
) -> dict[str, Any]:
    train_hash_cols = ["date", "ticker", *feature_cols, target_col, "target_date"]
    payload = {
        "model_name": model_name,
        "fold": int(fold_idx),
        "ticker": ticker,
        "training_scope": training_scope,
        "feature_cols": list(feature_cols),
        **_feature_set_metadata(feature_cols),
        "target_col": target_col,
        "params": dict(params),
        "target_purge_policy": TARGET_PURGE_POLICY,
        "n_train": int(len(train)),
        "n_train_before_purge": int(len(train_raw)),
        "n_train_purged": int(len(train_raw) - len(train)),
        "n_valid": int(len(valid)),
        "train_start": train["date"].min(),
        "train_end": train["date"].max(),
        "max_train_target_date": train["target_date"].max(),
        "validation_start": valid["date"].min(),
        "validation_end": valid["date"].max(),
        "train_data_hash": _dataframe_hash(train, [col for col in train_hash_cols if col in train.columns]),
    }
    payload.update(dict(metadata or {}))
    return payload


def _final_model_metadata(
    model_name: str,
    training_scope: str,
    feature_cols: list[str],
    target_col: str,
    params: dict[str, Any],
    train: pd.DataFrame,
    metadata: dict[str, Any] | None = None,
    ticker: str | None = None,
) -> dict[str, Any]:
    train_hash_cols = ["date", "ticker", *feature_cols, target_col, "target_date"]
    payload = {
        "model_name": model_name,
        "ticker": ticker,
        "training_scope": training_scope,
        **_feature_set_metadata(feature_cols),
        "target_col": target_col,
        "params": dict(params),
        "n_train": int(len(train)),
        "train_start": train["date"].min(),
        "train_end": train["date"].max(),
        "train_data_hash": _dataframe_hash(train, [col for col in train_hash_cols if col in train.columns]),
    }
    payload.update(dict(metadata or {}))
    return payload


def _model_metadata_matches(payload: dict[str, Any], expected_metadata: dict[str, Any]) -> bool:
    for key, expected in expected_metadata.items():
        if key not in payload:
            return False
        if not _metadata_values_match(payload[key], expected):
            return False
    return True


def _metadata_values_match(left: Any, right: Any) -> bool:
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            return False
        return all(_metadata_values_match(left[key], right[key]) for key in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _metadata_values_match(left_item, right_item)
            for left_item, right_item in zip(left, right)
        )
    if _is_datetime_like(left) or _is_datetime_like(right):
        return pd.Timestamp(left) == pd.Timestamp(right)
    return left == right


def _is_datetime_like(value: Any) -> bool:
    return isinstance(value, pd.Timestamp) or hasattr(value, "to_datetime64")


def _dataframe_hash(df: pd.DataFrame, columns: list[str]) -> str:
    if not columns:
        return ""
    hashes = pd.util.hash_pandas_object(df[columns], index=True).to_numpy(dtype="uint64", copy=False)
    return hashlib.sha256(hashes.tobytes()).hexdigest()


def safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._") or "ticker"


def _feature_set_metadata(feature_cols: list[str]) -> dict[str, Any]:
    return {
        "feature_count": int(len(feature_cols)),
        "feature_set_hash": _feature_set_hash(feature_cols),
        "feature_cols": list(feature_cols),
    }


def _feature_set_hash(feature_cols: list[str]) -> str:
    payload = "\n".join(map(str, feature_cols)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _prediction_frame(
    valid: pd.DataFrame,
    target_col: str,
    fold_idx: int,
    model_name: str,
    y_pred: np.ndarray,
    metadata: dict[str, Any] | None = None,
) -> pd.DataFrame:
    optional_cols = ["close", "future_close", "entry_date", "entry_open", "future_open", "target_date"]
    pred_cols = ["date", "ticker", *[col for col in optional_cols if col in valid.columns], target_col]
    pred = valid[pred_cols].copy()
    pred["fold"] = fold_idx
    pred["model_name"] = model_name
    pred["y_true"] = valid[target_col].to_numpy()
    pred["y_pred"] = y_pred
    for key, value in dict(metadata or {}).items():
        pred[key] = value
    return pred


def _first_available_ticker_train(
    model_df: pd.DataFrame,
    splits: list[dict[str, pd.DatetimeIndex]],
    ticker: str,
) -> pd.DataFrame:
    for split in splits:
        train_raw = model_df[model_df["date"].isin(split["train_dates"])].sort_values(["date", "ticker"])
        train = _purge_train_target_overlap(train_raw, split["validation_dates"].min())
        ticker_train = train[train["ticker"].astype(str) == ticker]
        if len(ticker_train) >= 5:
            return ticker_train
    return pd.DataFrame()


def _fold_metrics(
    pred: pd.DataFrame,
    split: dict[str, pd.DatetimeIndex],
    model_name: str,
    n_train: int,
    n_train_before_purge: int,
    ticker: str | None = None,
) -> dict[str, Any]:
    metrics = regression_metrics(pred["y_true"], pred["y_pred"])
    metrics.update(
        {
            "model_name": model_name,
            "fold": int(pred["fold"].iloc[0]),
            "ticker": ticker,
            "training_scope": "per_ticker" if ticker is not None else "global",
            "train_start": split["train_dates"].min(),
            "train_end": split["train_dates"].max(),
            "validation_start": split["validation_dates"].min(),
            "validation_end": split["validation_dates"].max(),
            "n_train": int(n_train),
            "n_train_before_purge": int(n_train_before_purge),
            "n_train_purged": int(n_train_before_purge - n_train),
            "n_valid": int(len(pred)),
        }
    )
    return metrics


def build_estimator(
    model_config: dict[str, Any],
    params: dict[str, Any] | None = None,
    random_state: int = 42,
):
    factory: Callable[..., ModelSpec] = model_config.get("estimator_factory", build_model)
    spec = factory(model_config["model_type"], random_state=random_state, params=params or {})
    if "needs_scaler" in model_config:
        spec.scale_features = bool(model_config["needs_scaler"])
    if "input_mode" in model_config:
        spec.input_mode = str(model_config["input_mode"])
    return make_pipeline(spec)


def expand_param_grid(search_params: dict[str, list[Any]], max_trials: int | None = None) -> list[dict[str, Any]]:
    if not search_params:
        return [{}]
    keys = list(search_params)
    values = [search_params[key] for key in keys]
    candidates = [dict(zip(keys, combo)) for combo in product(*values)]
    if max_trials is not None:
        return candidates[:max_trials]
    return candidates


def normalize_search_space(model_config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    search_space = model_config.get("search_space")
    if search_space is None:
        search_params = model_config.get("search_params", {})
        return {
            name: {"type": "categorical", "choices": list(values)}
            for name, values in dict(search_params).items()
            if list(values)
        }

    normalized: dict[str, dict[str, Any]] = {}
    for name, raw_spec in dict(search_space).items():
        if isinstance(raw_spec, list):
            normalized[name] = {"type": "categorical", "choices": raw_spec}
            continue
        if not isinstance(raw_spec, dict):
            raise TypeError(f"Search space for {name} must be a dict or list")
        spec = dict(raw_spec)
        spec_type = spec.get("type")
        if spec_type not in {"float", "int", "categorical"}:
            raise ValueError(f"Unsupported search space type for {name}: {spec_type}")
        if spec_type == "categorical":
            choices = list(spec.get("choices", []))
            if not choices:
                raise ValueError(f"Categorical search space for {name} must include choices")
            normalized[name] = {"type": "categorical", "choices": choices}
            continue

        if "low" not in spec or "high" not in spec:
            raise ValueError(f"Search space for {name} must include low and high")
        clean_spec = {"type": spec_type, "low": spec["low"], "high": spec["high"]}
        if "step" in spec:
            clean_spec["step"] = spec["step"]
        if spec.get("log"):
            clean_spec["log"] = True
        normalized[name] = clean_spec
    return normalized


def suggest_params(trial: Any, search_space: dict[str, dict[str, Any]]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for name, spec in search_space.items():
        spec_type = spec["type"]
        if spec_type == "categorical":
            params[name] = trial.suggest_categorical(name, list(spec["choices"]))
        elif spec_type == "int":
            kwargs = {"step": int(spec.get("step", 1))}
            if spec.get("log"):
                kwargs["log"] = True
            params[name] = trial.suggest_int(name, int(spec["low"]), int(spec["high"]), **kwargs)
        elif spec_type == "float":
            kwargs = {}
            if "step" in spec:
                kwargs["step"] = float(spec["step"])
            if spec.get("log"):
                kwargs["log"] = True
            params[name] = trial.suggest_float(name, float(spec["low"]), float(spec["high"]), **kwargs)
        else:
            raise ValueError(f"Unsupported search space type for {name}: {spec_type}")
    return params


def optuna_trials_payload(trials: list[Any]) -> list[dict[str, Any]]:
    payload = []
    for trial in trials:
        metrics = {
            key: value
            for key, value in trial.user_attrs.items()
            if key in METRIC_DIRECTIONS
        }
        payload.append(
            {
                "number": int(trial.number),
                "state": trial.state.name,
                "value": None if trial.value is None else float(trial.value),
                "params": dict(trial.params),
                "metrics": metrics,
            }
        )
    return payload


def _save_static_hyperparameter_fallback(
    model_config: dict[str, Any],
    metric: str,
    static_params: dict[str, Any],
    search_space: dict[str, dict[str, Any]],
    n_trials: int,
    optuna_n_jobs: int,
    cache_path: Path,
    reason: str,
    target_dates: pd.Series | None,
    target_col: str | None,
    execution_timing: str | None,
    cache_metadata: dict[str, Any] | None = None,
    inner_validation_start: pd.Timestamp | None = None,
    n_inner_train_before_purge: int | None = None,
    n_inner_train_after_purge: int | None = None,
    n_inner_valid: int | None = None,
) -> dict[str, Any]:
    best_params = dict(static_params)
    payload = _hyperparameter_payload(
        model_config["name"],
        metric,
        static_params,
        dict(model_config.get("search_params", {})),
        search_space,
        n_trials,
        optuna_n_jobs,
        STATIC_FALLBACK_TUNING_BACKEND,
        model_config.get("max_trials"),
        TARGET_PURGE_POLICY if target_dates is not None else None,
        target_col,
        execution_timing,
        best_params,
        [],
        [],
        best_trial_number=None,
        cache_metadata=cache_metadata,
    )
    payload["fallback_reason"] = reason
    if inner_validation_start is not None:
        payload["inner_validation_start"] = inner_validation_start
    if n_inner_train_before_purge is not None:
        payload["n_inner_train_before_purge"] = int(n_inner_train_before_purge)
    if n_inner_train_after_purge is not None:
        payload["n_inner_train_after_purge"] = int(n_inner_train_after_purge)
    if n_inner_valid is not None:
        payload["n_inner_valid"] = int(n_inner_valid)
    save_json(payload, cache_path)
    return best_params


def chronological_holdout(
    X: pd.DataFrame,
    y: pd.Series,
    feature_dates: pd.Series | None = None,
    validation_fraction: float = 0.2,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.Timestamp]:
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    split_at = int(len(X) * (1.0 - validation_fraction))
    if split_at <= 0 or split_at >= len(X):
        raise ValueError("Not enough rows for chronological holdout")
    if feature_dates is None:
        validation_start = X.iloc[split_at:].index.min()
    else:
        validation_start = pd.to_datetime(pd.Series(feature_dates).iloc[split_at:]).min()
    return X.iloc[:split_at], y.iloc[:split_at], X.iloc[split_at:], y.iloc[split_at:], validation_start


def summarize_metrics(
    metrics_df: pd.DataFrame,
    primary_metric: str = "directional_accuracy",
    group_cols: str | list[str] = "model_name",
) -> pd.DataFrame:
    metric_cols = ["mae", "rmse", "r2", "pearson", "spearman", "directional_accuracy"]
    group_columns = [group_cols] if isinstance(group_cols, str) else list(group_cols)
    summary = (
        metrics_df.groupby(group_columns)[metric_cols]
        .agg(["mean", "std"])
        .sort_values((primary_metric, "mean"), ascending=METRIC_DIRECTIONS.get(primary_metric, "min") == "min")
    )
    summary.columns = ["_".join(col).strip("_") for col in summary.columns]
    return summary.reset_index()


def _is_better(score: float, best_score: float, metric: str) -> bool:
    if not np.isfinite(score):
        return False
    if METRIC_DIRECTIONS.get(metric, "min") == "max":
        return score > best_score
    return score < best_score


def _hyperparameter_payload(
    model_name: str,
    metric: str,
    static_params: dict[str, Any],
    search_params: dict[str, list[Any]],
    search_space: dict[str, dict[str, Any]],
    n_trials: int,
    optuna_n_jobs: int,
    tuning_backend: str,
    max_trials: int | None,
    target_purge_policy: str | None,
    target_col: str | None,
    execution_timing: str | None,
    best_params: dict[str, Any],
    search_results: list[dict[str, Any]],
    trials: list[dict[str, Any]],
    best_score: float | None = None,
    best_trial_number: int | None = None,
    cache_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "model_name": model_name,
        "metric": metric,
        "tuning_backend": tuning_backend,
        "static_params": static_params,
        "search_params": search_params,
        "search_space": search_space,
        "n_trials": n_trials,
        "optuna_n_jobs": optuna_n_jobs,
        "max_trials": max_trials,
        "target_purge_policy": target_purge_policy,
        "best_params": best_params,
        "search_results": search_results,
        "trials": trials,
        "best_trial_number": best_trial_number,
    }
    if target_col is not None:
        payload["target_col"] = target_col
    if execution_timing is not None:
        payload["execution_timing"] = execution_timing
    if best_score is not None:
        payload["best_score"] = best_score
    if cache_metadata:
        payload.update(cache_metadata)
    return payload


def _hyperparameter_cache_matches(
    cached: dict[str, Any],
    model_config: dict[str, Any],
    metric: str,
    target_dates: pd.Series | None,
    target_col: str | None,
    execution_timing: str | None,
    search_space: dict[str, dict[str, Any]],
    tuning_backend: str,
    n_trials: int,
    cache_metadata: dict[str, Any] | None = None,
) -> bool:
    if target_dates is not None and cached.get("target_purge_policy") != TARGET_PURGE_POLICY:
        return False
    if target_dates is None and cached.get("target_purge_policy") is not None:
        return False
    if target_col is not None and cached.get("target_col") != target_col:
        return False
    if execution_timing is not None and cached.get("execution_timing") != execution_timing:
        return False
    if cached.get("metric") != metric:
        return False
    for key, value in dict(cache_metadata or {}).items():
        if cached.get(key) != value:
            return False
    expected_tuning_backends = {tuning_backend}
    if tuning_backend == "optuna":
        expected_tuning_backends.add(STATIC_FALLBACK_TUNING_BACKEND)
    return (
        cached.get("static_params") == dict(model_config.get("static_params", {}))
        and cached.get("search_space") == search_space
        and cached.get("tuning_backend") in expected_tuning_backends
        and cached.get("n_trials") == n_trials
    )


def _metrics_payload(metrics_df: pd.DataFrame) -> dict[str, Any]:
    numeric = metrics_df.select_dtypes(include="number").drop(columns=["fold"], errors="ignore")
    return {
        "fold_metrics": metrics_df.to_dict(orient="records"),
        "mean_metrics": numeric.mean().to_dict(),
        "std_metrics": numeric.std().to_dict(),
    }


def _ensure_target_date(model_df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    out = model_df.copy()
    if "target_date" in out.columns:
        out["target_date"] = pd.to_datetime(out["target_date"])
        return out

    horizon = _parse_target_horizon(target_col)
    out = out.sort_values(["ticker", "date"]).copy()
    shift = -(horizon + 1) if _target_execution_timing(target_col) == "next_open" else -horizon
    out["target_date"] = out.groupby("ticker", sort=False)["date"].shift(shift)
    return out


def _parse_target_horizon(target_col: str) -> int:
    horizon, _ = _parse_target_spec(target_col)
    return horizon


def _target_execution_timing(target_col: str) -> str:
    _, execution_timing = _parse_target_spec(target_col)
    return execution_timing


def _parse_target_spec(target_col: str) -> tuple[int, str]:
    parts = target_col.split("_")
    try:
        if parts[:2] != ["target", "return"] or len(parts) < 3:
            raise ValueError
        horizon = int(parts[2])
    except ValueError as exc:
        raise ValueError(f"Cannot infer target horizon from target column: {target_col}") from exc
    suffix = "_".join(parts[3:])
    if not suffix:
        return horizon, "close_to_close"
    if suffix == "next_open":
        return horizon, "next_open"
    raise ValueError(f"Cannot infer target execution timing from target column: {target_col}")


def _purge_train_target_overlap(train: pd.DataFrame, validation_start: pd.Timestamp) -> pd.DataFrame:
    if "target_date" not in train.columns:
        raise ValueError("target_date column is required for leakage-safe purging")
    validation_start = pd.Timestamp(validation_start)
    return train[pd.to_datetime(train["target_date"]) < validation_start].copy()
