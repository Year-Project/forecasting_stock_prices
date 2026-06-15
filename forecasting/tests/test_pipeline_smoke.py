import json

import joblib
import numpy as np
import pandas as pd

from stock_forecast.features import get_feature_columns, make_features
from stock_forecast.pipeline import get_or_search_best_params, run_walk_forward_training
from stock_forecast.splits import generate_walk_forward_splits
from stock_forecast.targets import make_future_return_target


class ConstantEstimator:
    def __init__(self, value: float):
        self.value = value

    def predict(self, X):
        return np.full(len(X), self.value)


def _synthetic_dataset(horizon: int = 2):
    rows = []
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    for ticker, offset in [("A", 0.0), ("B", 5.0)]:
        for idx, date in enumerate(dates):
            close = 100.0 + offset + idx * 0.5
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "open": close - 0.2,
                    "high": close + 0.5,
                    "low": close - 0.5,
                    "close": close,
                    "volume": 1000 + idx,
                }
            )
    df = pd.DataFrame(rows)
    df = make_features(df, return_lags=[1, 2], rolling_windows=[2, 3, 5], use_technical_indicators=False)
    df = make_future_return_target(df, horizon=horizon, execution_timing="next_open")
    feature_cols = get_feature_columns(df)
    target_col = f"target_return_{horizon}_next_open"
    assert "target_date" not in feature_cols
    assert "entry_open" not in feature_cols
    assert "future_open" not in feature_cols
    return df.dropna(subset=[*feature_cols, target_col]).reset_index(drop=True), feature_cols, target_col


def _ridge_config(**overrides):
    config = {
        "name": "ridge",
        "model_type": "ridge",
        "static_params": {},
        "search_space": {"alpha": {"type": "categorical", "choices": [0.1, 1.0]}},
        "n_trials": 2,
        "optuna_n_jobs": 2,
        "needs_scaler": True,
    }
    config.update(overrides)
    return config


def _naive_config():
    return {
        "name": "naive_persistence",
        "model_type": "naive_persistence",
        "static_params": {},
        "search_params": {},
        "needs_scaler": False,
    }


def test_walk_forward_training_smoke(tmp_path):
    df, feature_cols, target_col = _synthetic_dataset()
    splits = generate_walk_forward_splits(df, train_window=25, validation_window=10, step=10)
    configs = [_ridge_config(), _naive_config()]

    predictions, metrics, best_params = run_walk_forward_training(
        df,
        feature_cols,
        target_col,
        splits[:2],
        configs,
        tmp_path,
        force_retrain=True,
        run_metadata={"horizon_name": "week", "horizon": 2},
    )

    assert set(predictions["model_name"]) == {"ridge", "naive_persistence"}
    assert set(metrics["model_name"]) == {"ridge", "naive_persistence"}
    assert {"entry_date", "entry_open", "future_open", "target_date"}.issubset(predictions.columns)
    assert predictions["y_true"].equals(predictions[target_col])
    assert set(predictions["horizon_name"]) == {"week"}
    assert set(metrics["horizon_name"]) == {"week"}
    assert (metrics["n_train_purged"] > 0).all()
    assert "ridge" in best_params
    assert best_params["naive_persistence"] == {}
    assert (tmp_path / "hyperparams" / "ridge.json").exists()
    assert (tmp_path / "predictions" / "ridge_oof.parquet").exists() or (
        tmp_path / "predictions" / "ridge_oof.csv"
    ).exists()
    assert (tmp_path / "reports" / "all_model_ticker_metrics.parquet").exists() or (
        tmp_path / "reports" / "all_model_ticker_metrics.csv"
    ).exists()


def test_walk_forward_training_can_fit_one_model_per_ticker(tmp_path):
    df, feature_cols, target_col = _synthetic_dataset()
    splits = generate_walk_forward_splits(df, train_window=25, validation_window=10, step=10)
    configs = [_ridge_config(), _naive_config()]

    predictions, metrics, best_params = run_walk_forward_training(
        df,
        feature_cols,
        target_col,
        splits[:2],
        configs,
        tmp_path,
        force_retrain=True,
        train_by_ticker=True,
    )

    assert set(predictions["ticker"]) == {"A", "B"}
    assert set(metrics["training_scope"]) == {"per_ticker", "per_ticker_aggregate"}
    assert set(best_params["ridge"]) == {"A", "B"}
    assert best_params["naive_persistence"]["A"] == {}
    assert (tmp_path / "hyperparams" / "ridge" / "A.json").exists()
    assert (tmp_path / "models" / "ridge" / "A" / "fold_0.pkl").exists()
    assert (tmp_path / "models" / "ridge" / "A" / "final.pkl").exists()
    assert (tmp_path / "models" / "ridge" / "B" / "final.pkl").exists()


def test_hyperparameter_cache_is_reused_when_file_exists(tmp_path):
    df, feature_cols, target_col = _synthetic_dataset()
    config = _ridge_config()
    first = get_or_search_best_params(
        config,
        df[feature_cols],
        df[target_col],
        tmp_path,
        force_retrain=True,
        target_col=target_col,
        execution_timing="next_open",
        cache_metadata={"horizon_name": "week", "horizon": 2},
    )
    second = get_or_search_best_params(
        config,
        df[feature_cols],
        df[target_col],
        tmp_path,
        force_retrain=False,
        target_col=target_col,
        execution_timing="next_open",
        cache_metadata={"horizon_name": "week", "horizon": 2},
    )
    payload = json.loads((tmp_path / "ridge.json").read_text())
    assert payload["tuning_backend"] == "optuna"
    assert payload["n_trials"] == 2
    assert payload["optuna_n_jobs"] == 2
    assert payload["search_space"] == config["search_space"]
    assert payload["best_trial_number"] is not None
    assert len(payload["trials"]) == 2
    assert payload["horizon_name"] == "week"

    changed_config = config | {"search_space": {"alpha": {"type": "categorical", "choices": [999.0]}}, "n_trials": 1}
    changed = get_or_search_best_params(
        changed_config,
        df[feature_cols],
        df[target_col],
        tmp_path,
        force_retrain=False,
        target_col=target_col,
        execution_timing="next_open",
        cache_metadata={"horizon_name": "week", "horizon": 2},
    )
    assert second == first
    assert changed == {"alpha": 999.0}


def test_hyperparameter_cache_is_not_reused_when_feature_set_changes(tmp_path):
    df, feature_cols, target_col = _synthetic_dataset()
    config = _ridge_config(search_space={"alpha": {"type": "categorical", "choices": [0.1]}}, n_trials=1)
    first_cols = feature_cols[:3]
    second_cols = feature_cols[:4]

    get_or_search_best_params(
        config,
        df[first_cols],
        df[target_col],
        tmp_path,
        force_retrain=True,
        target_col=target_col,
        execution_timing="next_open",
        cache_metadata={"horizon_name": "week", "horizon": 2},
    )
    first_payload = json.loads((tmp_path / "ridge.json").read_text())

    get_or_search_best_params(
        config,
        df[second_cols],
        df[target_col],
        tmp_path,
        force_retrain=False,
        target_col=target_col,
        execution_timing="next_open",
        cache_metadata={"horizon_name": "week", "horizon": 2},
    )
    second_payload = json.loads((tmp_path / "ridge.json").read_text())

    assert first_payload["feature_count"] == 3
    assert first_payload["feature_cols"] == first_cols
    assert second_payload["feature_count"] == 4
    assert second_payload["feature_cols"] == second_cols
    assert second_payload["feature_set_hash"] != first_payload["feature_set_hash"]


def test_stale_best_param_file_is_not_reused_without_matching_metadata(tmp_path):
    df, feature_cols, target_col = _synthetic_dataset()
    config = _ridge_config()
    cache_path = tmp_path / "ridge.json"
    cache_path.write_text(
        json.dumps(
            {
                "model_name": "ridge",
                "metric": "rmse",
                "static_params": {},
                "search_params": {"alpha": [0.1, 1.0]},
                "max_trials": None,
                "target_purge_policy": None,
                "best_params": {"alpha": 123.0},
                "search_results": [],
            }
        )
    )

    best = get_or_search_best_params(
        config,
        df[feature_cols],
        df[target_col],
        tmp_path,
        force_retrain=False,
        target_col=target_col,
        execution_timing="next_open",
    )

    cached = json.loads(cache_path.read_text())
    assert best != {"alpha": 123.0}
    assert cached["target_col"] == target_col
    assert cached["execution_timing"] == "next_open"
    assert cached["tuning_backend"] == "optuna"


def test_old_search_params_are_converted_to_optuna_categorical_space(tmp_path):
    df, feature_cols, target_col = _synthetic_dataset()
    config = {
        "name": "ridge",
        "model_type": "ridge",
        "static_params": {},
        "search_params": {"alpha": [0.1]},
        "n_trials": 1,
        "needs_scaler": True,
    }

    best = get_or_search_best_params(
        config,
        df[feature_cols],
        df[target_col],
        tmp_path,
        force_retrain=True,
        target_col=target_col,
        execution_timing="next_open",
    )
    payload = json.loads((tmp_path / "ridge.json").read_text())

    assert best == {"alpha": 0.1}
    assert payload["search_space"] == {"alpha": {"type": "categorical", "choices": [0.1]}}
    assert payload["tuning_backend"] == "optuna"


def test_hyperparameter_search_uses_static_fallback_when_purged_split_is_empty(tmp_path):
    dates = pd.date_range("2024-01-01", periods=16, freq="D")
    X = pd.DataFrame({"feature": np.arange(len(dates), dtype=float)})
    y = pd.Series(np.linspace(0.0, 1.0, len(dates)))
    target_dates = dates + pd.Timedelta(days=30)
    config = _ridge_config(static_params={"alpha": 2.0})

    best = get_or_search_best_params(
        config,
        X,
        y,
        tmp_path,
        force_retrain=True,
        feature_dates=pd.Series(dates),
        target_dates=pd.Series(target_dates),
        target_col="target_return_21_next_open",
        execution_timing="next_open",
    )

    payload = json.loads((tmp_path / "ridge.json").read_text())
    assert best == {"alpha": 2.0}
    assert payload["tuning_backend"] == "static_fallback"
    assert payload["fallback_reason"] == "purged_hyperparameter_train_split_empty"
    assert payload["n_inner_train_before_purge"] == 12
    assert payload["n_inner_train_after_purge"] == 0
    assert payload["target_purge_policy"] == "target_date_before_validation_start"


def test_walk_forward_training_keeps_horizon_artifacts_isolated(tmp_path):
    configs = [_ridge_config(n_trials=1, optuna_n_jobs=1)]

    for horizon_name, horizon in [("week", 2), ("month", 4)]:
        df, feature_cols, target_col = _synthetic_dataset(horizon=horizon)
        splits = generate_walk_forward_splits(df, train_window=25, validation_window=10, step=10)
        run_walk_forward_training(
            df,
            feature_cols,
            target_col,
            splits[:1],
            configs,
            tmp_path / "horizons" / horizon_name,
            force_retrain=True,
            train_by_ticker=True,
            run_metadata={"horizon_name": horizon_name, "horizon": horizon},
        )

    week_payload = json.loads((tmp_path / "horizons" / "week" / "hyperparams" / "ridge" / "A.json").read_text())
    month_payload = json.loads((tmp_path / "horizons" / "month" / "hyperparams" / "ridge" / "A.json").read_text())
    week_predictions = tmp_path / "horizons" / "week" / "predictions" / "all_oof_predictions.parquet"
    month_predictions = tmp_path / "horizons" / "month" / "predictions" / "all_oof_predictions.parquet"

    assert week_payload["horizon_name"] == "week"
    assert week_payload["target_col"] == "target_return_2_next_open"
    assert month_payload["horizon_name"] == "month"
    assert month_payload["target_col"] == "target_return_4_next_open"
    assert week_predictions.exists() or week_predictions.with_suffix(".csv").exists()
    assert month_predictions.exists() or month_predictions.with_suffix(".csv").exists()


def test_walk_forward_training_reuses_existing_fold_and_final_models(tmp_path):
    df, feature_cols, target_col = _synthetic_dataset()
    splits = generate_walk_forward_splits(df, train_window=25, validation_window=10, step=10)
    configs = [_ridge_config(n_trials=1, optuna_n_jobs=1)]

    run_walk_forward_training(
        df,
        feature_cols,
        target_col,
        splits[:1],
        configs,
        tmp_path,
        force_retrain=True,
    )

    fold_path = tmp_path / "models" / "ridge" / "fold_0.pkl"
    final_path = tmp_path / "models" / "ridge" / "final.pkl"
    fold_payload = joblib.load(fold_path)
    fold_payload["estimator"] = ConstantEstimator(123.0)
    joblib.dump(fold_payload, fold_path)
    final_payload = joblib.load(final_path)
    final_payload["sentinel"] = "keep-existing-final"
    joblib.dump(final_payload, final_path)

    predictions, _, _ = run_walk_forward_training(
        df,
        feature_cols,
        target_col,
        splits[:1],
        configs,
        tmp_path,
        force_retrain=False,
    )

    reloaded_final = joblib.load(final_path)
    assert np.allclose(predictions["y_pred"], 123.0)
    assert reloaded_final["sentinel"] == "keep-existing-final"


def test_walk_forward_training_retrains_stale_fold_model_metadata(tmp_path):
    df, feature_cols, target_col = _synthetic_dataset()
    splits = generate_walk_forward_splits(df, train_window=25, validation_window=10, step=10)
    configs = [_ridge_config(n_trials=1, optuna_n_jobs=1)]

    run_walk_forward_training(
        df,
        feature_cols,
        target_col,
        splits[:1],
        configs,
        tmp_path,
        force_retrain=True,
    )

    fold_path = tmp_path / "models" / "ridge" / "fold_0.pkl"
    fold_payload = joblib.load(fold_path)
    original_train_hash = fold_payload["train_data_hash"]
    fold_payload["estimator"] = ConstantEstimator(999.0)
    fold_payload["validation_start"] = pd.Timestamp("2099-01-01")
    joblib.dump(fold_payload, fold_path)

    predictions, _, _ = run_walk_forward_training(
        df,
        feature_cols,
        target_col,
        splits[:1],
        configs,
        tmp_path,
        force_retrain=False,
    )

    reloaded_payload = joblib.load(fold_path)
    assert not np.allclose(predictions["y_pred"], 999.0)
    assert reloaded_payload["validation_start"] != pd.Timestamp("2099-01-01")
    assert reloaded_payload["train_data_hash"] == original_train_hash


def test_walk_forward_training_retrains_stale_final_model_feature_metadata(tmp_path):
    df, feature_cols, target_col = _synthetic_dataset()
    splits = generate_walk_forward_splits(df, train_window=25, validation_window=10, step=10)
    configs = [_ridge_config(n_trials=1, optuna_n_jobs=1)]
    first_cols = feature_cols[:3]
    second_cols = feature_cols[:4]

    run_walk_forward_training(
        df,
        first_cols,
        target_col,
        splits[:1],
        configs,
        tmp_path,
        force_retrain=True,
    )

    final_path = tmp_path / "models" / "ridge" / "final.pkl"
    final_payload = joblib.load(final_path)
    final_payload["sentinel"] = "stale-feature-final"
    joblib.dump(final_payload, final_path)

    run_walk_forward_training(
        df,
        second_cols,
        target_col,
        splits[:1],
        configs,
        tmp_path,
        force_retrain=False,
    )

    reloaded_payload = joblib.load(final_path)
    assert "sentinel" not in reloaded_payload
    assert reloaded_payload["feature_count"] == 4
    assert reloaded_payload["feature_cols"] == second_cols
