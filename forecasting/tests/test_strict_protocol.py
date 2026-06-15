import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from stock_forecast.features import get_feature_columns, make_features
from stock_forecast.strict_protocol import (
    _limited_history_config,
    _post_selection_params,
    make_strict_outer_splits,
    run_strict_global_lstm_protocol,
    run_strict_per_ticker_protocol,
)
from stock_forecast.targets import make_future_return_target


def _synthetic_dataset(horizon: int = 5, periods: int = 220):
    rows = []
    dates = pd.date_range("2024-01-01", periods=periods, freq="D")
    for ticker, offset in [("A", 0.0), ("B", 7.0)]:
        for idx, date in enumerate(dates):
            close = 100.0 + offset + idx * 0.2 + np.sin(idx / 7.0)
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "open": close - 0.1,
                    "high": close + 0.4,
                    "low": close - 0.4,
                    "close": close,
                    "volume": 1000 + idx,
                }
            )
    df = pd.DataFrame(rows)
    df = make_features(df, return_lags=[1, 2, 3], rolling_windows=[2, 3, 5], use_technical_indicators=False)
    df = make_future_return_target(df, horizon=horizon, execution_timing="next_open")
    feature_cols = get_feature_columns(df)
    target_col = f"target_return_{horizon}_next_open"
    return df.dropna(subset=[*feature_cols, target_col]).reset_index(drop=True), feature_cols, target_col


def _ridge_config():
    return {
        "name": "ridge",
        "model_type": "ridge",
        "static_params": {},
        "search_space": {"alpha": {"type": "categorical", "choices": [0.1, 1.0]}},
        "n_trials": 2,
        "optuna_n_jobs": 1,
        "needs_scaler": True,
    }


def _naive_config():
    return {
        "name": "naive_persistence",
        "model_type": "naive_persistence",
        "static_params": {},
        "search_space": {},
        "needs_scaler": False,
    }


def _tiny_lstm_config(feature_cols):
    return {
        "name": "lstm",
        "model_type": "lstm",
        "static_params": {
            "lookback": 5,
            "hidden_size": 8,
            "num_layers": 1,
            "input_projection_size": 0,
            "head_dropout": 0.0,
            "learning_rate": 1e-3,
            "weight_decay": 1e-5,
            "batch_size": 8,
            "max_epochs": 2,
            "patience": 1,
            "device": "cpu",
        },
        "search_space": {},
        "input_mode": "full_frame",
        "feature_cols": feature_cols,
        "needs_scaler": False,
    }


def _tiny_global_lstm_config(feature_cols):
    config = _tiny_lstm_config(feature_cols)
    config["name"] = "global_lstm"
    config["static_params"] = {
        **config["static_params"],
        "max_epochs": 1,
        "target_normalization": "per_ticker",
        "balanced_ticker_sampling": True,
    }
    return config


def test_strict_outer_split_boundaries_and_purge_inputs():
    df, _, target_col = _synthetic_dataset(periods=260)

    splits = make_strict_outer_splits(
        df,
        target_col=target_col,
        validation_rows=30,
        test_rows=30,
        mature_min_rows=180,
        min_train_rows=80,
    )

    assert set(splits["status"]) == {"ok"}
    assert (splits["n_validation"] == 30).all()
    assert (splits["n_test"] == 30).all()
    assert (pd.to_datetime(splits["train_end"]) < pd.to_datetime(splits["validation_start"])).all()
    assert (pd.to_datetime(splits["validation_end"]) < pd.to_datetime(splits["test_start"])).all()


def test_strict_outer_split_caps_training_history_to_recent_window():
    df, _, target_col = _synthetic_dataset(periods=320)

    splits = make_strict_outer_splits(
        df,
        target_col=target_col,
        validation_rows=30,
        test_rows=30,
        mature_min_rows=180,
        min_train_rows=80,
        max_train_rows=90,
    )

    assert set(splits["status"]) == {"ok"}
    assert (splits["n_train_available"] > splits["n_train"]).all()
    assert (splits["n_train"] == 90).all()
    assert (splits["n_refit"] == 90).all()
    assert (splits["max_train_rows"] == 90).all()
    first_dates = pd.to_datetime(df.groupby("ticker")["date"].min()).sort_index().to_numpy()
    assert (pd.to_datetime(splits.sort_values("ticker")["train_start"]).to_numpy() > first_dates).all()
    assert (pd.to_datetime(splits["train_end"]) < pd.to_datetime(splits["validation_start"])).all()
    assert (pd.to_datetime(splits["refit_end"]) < pd.to_datetime(splits["test_start"])).all()


def test_strict_protocol_writes_test_only_artifacts_and_final_models(tmp_path):
    df, feature_cols, target_col = _synthetic_dataset(periods=180)
    result = run_strict_per_ticker_protocol(
        df,
        feature_cols,
        target_col,
        [_ridge_config(), _naive_config()],
        tmp_path,
        force_retrain=True,
        run_metadata={"horizon_name": "week", "horizon": 5},
        validation_rows=25,
        test_rows=25,
        mature_min_rows=120,
        min_train_rows=70,
        inner_max_folds=2,
        inner_min_train_rows=40,
        inner_validation_window=15,
    )

    reports_dir = tmp_path / "strict_protocol" / "reports"
    assert (reports_dir / "test_predictions.parquet").exists()
    assert (reports_dir / "validation_predictions.parquet").exists()
    assert (reports_dir / "validation_model_ranking.parquet").exists()
    assert (reports_dir / "test_prediction_metrics.parquet").exists()
    assert not (tmp_path / "predictions" / "all_oof_predictions.parquet").exists()

    validation_predictions = result["validation_predictions"]
    test_predictions = result["test_predictions"]
    assert set(validation_predictions["split_role"]) == {"validation"}
    assert set(test_predictions["split_role"]) == {"test"}
    assert set(test_predictions["model_name"]) == {"ridge", "naive_persistence"}
    assert set(test_predictions["ticker"]) == {"A", "B"}

    split_bounds = result["outer_splits"].set_index("ticker")
    for ticker, group in validation_predictions.groupby("ticker"):
        bounds = split_bounds.loc[ticker]
        assert group["date"].min() >= bounds["validation_start"]
        assert group["date"].max() <= bounds["validation_end"]
    for ticker, group in test_predictions.groupby("ticker"):
        bounds = split_bounds.loc[ticker]
        assert group["date"].min() >= bounds["test_start"]
        assert group["date"].max() <= bounds["test_end"]
        assert (pd.to_datetime(group["max_train_target_date"]) < bounds["test_start"]).all()

    for model_name in ["ridge", "naive_persistence"]:
        for ticker in ["A", "B"]:
            path = tmp_path / "strict_protocol" / "models" / model_name / ticker / "final.pkl"
            payload = joblib.load(path)
            assert payload["training_scope"] == "per_ticker_strict_final"
            assert payload["ticker"] == ticker
            assert pd.Timestamp(payload["max_train_target_date"]) < split_bounds.loc[ticker, "test_start"]


def test_strict_protocol_signal_reports_are_test_only(tmp_path):
    df, feature_cols, target_col = _synthetic_dataset(periods=180)
    result = run_strict_per_ticker_protocol(
        df,
        feature_cols,
        target_col,
        [_naive_config()],
        tmp_path,
        force_retrain=True,
        run_metadata={"horizon_name": "week", "horizon": 5},
        validation_rows=25,
        test_rows=25,
        mature_min_rows=120,
        min_train_rows=70,
        inner_max_folds=2,
        inner_min_train_rows=40,
        inner_validation_window=15,
    )

    equity = result["test_signal_equity"]
    metrics = result["test_signal_metrics"]
    assert set(equity["signal_mode"]) == {"overlapping_tranches", "non_overlapping"}
    assert set(metrics["signal_mode"]) == {"overlapping_tranches", "non_overlapping"}

    test_signal_dates = set(pd.to_datetime(result["test_predictions"]["date"]))
    assert set(pd.to_datetime(equity["signal_date"])).issubset(test_signal_dates)


def test_strict_protocol_supports_full_frame_lstm_model(tmp_path):
    pytest.importorskip("torch")
    df, feature_cols, target_col = _synthetic_dataset(periods=180)
    result = run_strict_per_ticker_protocol(
        df,
        feature_cols,
        target_col,
        [_naive_config(), _tiny_lstm_config(feature_cols)],
        tmp_path,
        force_retrain=True,
        run_metadata={"horizon_name": "week", "horizon": 5},
        validation_rows=18,
        test_rows=18,
        mature_min_rows=90,
        min_train_rows=55,
        inner_max_folds=1,
        inner_min_train_rows=35,
        inner_validation_window=12,
    )

    assert {"naive_persistence", "lstm"} == set(result["test_predictions"]["model_name"])
    assert np.isfinite(result["test_predictions"]["y_pred"]).all()
    assert bool(result["leakage_audit"]["passed"].all())
    for ticker in ["A", "B"]:
        path = tmp_path / "strict_protocol" / "models" / "lstm" / ticker / "final.pkl"
        assert path.exists()
        payload = joblib.load(path)
        assert payload["training_scope"] == "per_ticker_strict_final"
        assert payload["feature_cols"] == feature_cols


def test_strict_global_lstm_protocol_writes_one_global_model_and_passes_audit(tmp_path):
    pytest.importorskip("torch")
    df, feature_cols, target_col = _synthetic_dataset(periods=150)
    result = run_strict_global_lstm_protocol(
        df,
        feature_cols,
        target_col,
        [_tiny_global_lstm_config(feature_cols)],
        tmp_path,
        force_retrain=True,
        run_metadata={"horizon_name": "week", "horizon": 5},
        validation_rows=15,
        test_rows=15,
        mature_min_rows=100,
        min_train_rows=50,
        max_train_rows=80,
        inner_max_folds=1,
        inner_min_train_rows=40,
        inner_validation_window=10,
        threshold_grid=[0.0],
        min_validation_trades=1,
    )

    assert set(result["test_predictions"]["model_name"]) == {"global_lstm"}
    assert set(result["test_predictions"]["ticker"]) == {"A", "B"}
    assert not result["test_panel_signal_metrics"].empty
    assert bool(result["leakage_audit"]["passed"].all())

    path = tmp_path / "strict_protocol" / "models" / "global_lstm" / "final.pkl"
    payload = joblib.load(path)
    assert payload["training_scope"] == "global_lstm_strict_final"
    assert payload["ticker"] == "__global__"
    assert pd.Timestamp(payload["max_train_target_date"]) < result["global_test_start"]


def test_strict_limited_history_config_overrides_search_space_and_trials():
    config = {
        "name": "lstm",
        "model_type": "lstm",
        "static_params": {"max_epochs": 150},
        "limited_history_static_params": {"max_epochs": 80},
        "search_space": {"lookback": {"type": "categorical", "choices": [20, 40, 60, 90, 126]}},
        "limited_history_search_space": {"lookback": {"type": "categorical", "choices": [20, 40, 60]}},
        "n_trials": 60,
        "limited_history_n_trials": 20,
        "post_selection_static_params": {"ensemble_seeds": [1, 7, 21, 42, 101]},
    }

    limited = _limited_history_config(config, limited_history=True, limited_history_n_trials=16)
    mature = _limited_history_config(config, limited_history=False, limited_history_n_trials=16)

    assert limited["search_space"]["lookback"]["choices"] == [20, 40, 60]
    assert limited["static_params"] == {"max_epochs": 80}
    assert limited["n_trials"] == 20
    assert mature["search_space"]["lookback"]["choices"] == [20, 40, 60, 90, 126]
    assert mature["n_trials"] == 60
    assert _post_selection_params(limited, {"lookback": 40}) == {
        "lookback": 40,
        "ensemble_seeds": [1, 7, 21, 42, 101],
    }


def test_model_comparison_notebook_uses_strict_artifacts_and_full_metric_comparison():
    notebook_json = json.loads(Path("forecasting/notebooks/03_model_comparison.ipynb").read_text())
    notebook = "\n".join("".join(cell.get("source", [])) for cell in notebook_json["cells"])

    assert "strict_protocol" in notebook
    assert 'PRIMARY_METRIC = "directional_accuracy"' in notebook
    assert "test_predictions.parquet" in notebook
    assert "validation_signal_metrics.parquet" in notebook
    assert "TABLE_COMPARISON_SOURCE" in notebook
    assert "LSTM_ONLY_ARTIFACT_NAME" in notebook
    assert 'LSTM_ONLY_ARTIFACT_NAME / "strict_protocol" / "reports"' in notebook
    assert "Run notebooks/02b_lstm_forecasting.ipynb first" in notebook
    assert "global_lstm/strict_protocol/reports" in notebook
    assert "GLOBAL_LSTM_ARTIFACT_NAME" in notebook
    assert "comparison_source" in notebook
    assert "test_panel_signal_metrics.parquet" in notebook
    assert "comparison_test_signal_metrics.parquet" in notebook
    assert "comparison_test_panel_signal_metrics.parquet" in notebook
    assert "validation_model_metric_comparison.parquet" in notebook
    assert "Validation-vs-test model metric comparison" in notebook
    assert "ensure_prediction_error_metrics" in notebook
    assert '"mse": "min"' in notebook
    assert '"smape": "min"' in notebook
    assert '"validation_mse"' in notebook
    assert '"validation_smape"' in notebook
    assert '"test_mse"' in notebook
    assert '"test_smape"' in notebook
    assert "build_all_horizon_test_metrics_table" in notebook
    assert "all_horizon_test_metrics.parquet" in notebook
    assert "Final Model Comparison Ranking" in notebook
    assert "build_final_model_comparison_ranking" in notebook
    assert "final_model_comparison_ranking.parquet" in notebook
    assert "model_rank" in notebook
    assert '("test_sharpe", False)' in notebook
    assert '("mae", True)' in notebook
    assert '("smape", True)' in notebook
    assert 'for metric_col in ["mae", "mse", "rmse", "smape"]' in notebook
    assert '"test_n_obs"' in notebook
    assert '"mae"' in notebook
    assert '"smape"' in notebook
    assert "All horizon, stock, and model metrics on test data sample" in notebook
    assert '"display.max_rows", None' in notebook
    assert '"display.max_columns", None' in notebook
    assert "## Test Predicted Vs Actual And Distributions For Each Ticker" in notebook
    assert 'test_predictions = result["test_predictions"]' in notebook
    assert "_test_predicted_vs_actual_by_date.png" in notebook
    assert "_test_predicted_vs_actual_scatter.png" in notebook
    assert "_test_prediction_distribution.png" in notebook
    assert "all_oof_predictions" not in notebook
    assert "run_per_ticker_signal_backtest" not in notebook
    assert "MIN_VALIDATION_TRADES" not in notebook
    assert "MAX_VALIDATION_DRAWDOWN" not in notebook
    assert "TURNOVER_PENALTY_WEIGHT" not in notebook
    assert "validation_constraints_pass" not in notebook


def test_table_model_forecasting_notebook_uses_directional_accuracy_selection():
    notebook = Path("forecasting/notebooks/02_table_model_forecasting.ipynb").read_text()

    assert 'PRIMARY_METRIC = \\"directional_accuracy\\"' in notebook
    assert 'primary_metric=PRIMARY_METRIC' in notebook
    assert 'PRIMARY_METRIC = \\"rmse\\"' not in notebook
    assert '\\"model_type\\": \\"lstm\\"' not in notebook
    assert '\\"input_mode\\": \\"full_frame\\"' not in notebook
    assert "LSTM_N_TRIALS" not in notebook
    assert "make_lstm_config" not in notebook
    assert "01b_lstm_eda.ipynb" not in notebook
    assert "lstm_only" not in notebook


def test_lstm_notebook_and_report_sources_are_registered():
    lstm_eda = Path("forecasting/notebooks/01b_lstm_eda.ipynb").read_text()
    global_lstm_eda = Path("forecasting/notebooks/01c_global_lstm_eda.ipynb").read_text()
    training = Path("forecasting/notebooks/02_table_model_forecasting.ipynb").read_text()
    lstm_only_training = Path("forecasting/notebooks/02b_lstm_forecasting.ipynb").read_text()
    global_lstm_training = Path("forecasting/notebooks/02c_global_lstm_forecasting.ipynb").read_text()
    comparison = Path("forecasting/notebooks/03_model_comparison.ipynb").read_text()

    assert "make_lstm_features" in lstm_eda
    assert "LSTM_DATA_DIR" in lstm_eda
    assert "sequence_diagnostics.json" in lstm_eda
    assert '\\"model_type\\": \\"lstm\\"' not in training
    assert '\\"input_mode\\": \\"full_frame\\"' not in training
    assert "LSTM_N_TRIALS" not in training
    assert "01b_lstm_eda.ipynb" not in training
    assert "02b_lstm_forecasting" in lstm_only_training
    assert '\\"model_type\\": \\"lstm\\"' in lstm_only_training
    assert '\\"input_mode\\": \\"full_frame\\"' in lstm_only_training
    assert "LSTM_N_TRIALS" in lstm_only_training
    assert "01b_lstm_eda.ipynb" in lstm_only_training
    assert "MERGE_LSTM_INTO_MAIN_STRICT_REPORTS" not in lstm_only_training
    assert "LSTM_ONLY_ARTIFACT_NAME" in lstm_only_training
    assert "lstm_only" in lstm_only_training
    assert "merge_lstm_into_main" not in lstm_only_training
    assert "_strict_metrics" not in lstm_only_training
    assert "_validation_ranking" not in lstm_only_training
    assert "FORCE_RETRAIN" in lstm_only_training
    assert "GLOBAL_LSTM_DATA_DIR" in global_lstm_eda
    assert "ticker_onehot_columns" in global_lstm_eda
    assert "global_stationary" in global_lstm_eda
    assert "global_sequence_diagnostics.json" in global_lstm_eda
    assert "run_strict_global_lstm_protocol" in global_lstm_training
    assert "validation_threshold_search" in global_lstm_training
    assert "test_panel_signal_metrics" in global_lstm_training
    assert "02c_global_lstm_forecasting" in global_lstm_training
    assert "lstm_hyperparameter_summary" in comparison
    assert "LSTM Hyperparameter Summary" in comparison
    assert "TABLE_COMPARISON_SOURCE" in comparison
    assert "LSTM_ONLY_ARTIFACT_NAME" in comparison
    assert "lstm_only" in comparison
    assert "global_lstm" in comparison
    assert "test_panel_signal_metrics" in comparison
