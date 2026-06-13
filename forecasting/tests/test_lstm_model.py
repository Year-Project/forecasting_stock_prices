import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

from stock_forecast.lstm_features import get_lstm_feature_columns, make_lstm_features
from stock_forecast.models.lstm import (
    LSTMReturnRegressor,
    _chronological_sequence_split_by_date,
    build_lstm_sequence_samples,
)


def _ohlcv_frame(periods: int = 80) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=periods, freq="D")
    close = 100.0 + np.arange(periods) * 0.1 + np.sin(np.arange(periods) / 5.0)
    return pd.DataFrame(
        {
            "date": dates,
            "ticker": "A",
            "open": close - 0.2,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": 1000 + np.arange(periods) * 3,
        }
    )


def test_lstm_features_do_not_depend_on_future_rows():
    base = _ohlcv_frame()
    changed_future = base.copy()
    changed_future.loc[70, ["open", "high", "low", "close", "volume"]] = [300.0, 320.0, 280.0, 310.0, 999999.0]

    before = make_lstm_features(base)
    after = make_lstm_features(changed_future)
    feature_cols = get_lstm_feature_columns(before)
    check_date = pd.Timestamp("2024-02-20")
    before_row = before.loc[before["date"] == check_date, feature_cols].reset_index(drop=True)
    after_row = after.loc[after["date"] == check_date, feature_cols].reset_index(drop=True)

    pd.testing.assert_frame_equal(before_row, after_row)


def test_lstm_sequence_builder_never_crosses_ticker_boundaries():
    rows = []
    for ticker, offset in [("A", 0.0), ("B", 100.0)]:
        for idx in range(5):
            rows.append(
                {
                    "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=idx),
                    "ticker": ticker,
                    "feature": offset + idx,
                    "target": idx / 10.0,
                }
            )
    frame = pd.DataFrame(rows)
    samples = build_lstm_sequence_samples(frame, ["feature"], lookback=3, target_col="target")

    assert samples.x.shape == (6, 3, 1)
    assert samples.x[0, :, 0].tolist() == [0.0, 1.0, 2.0]
    assert samples.x[3, :, 0].tolist() == [100.0, 101.0, 102.0]
    assert samples.ticker == ["A", "A", "A", "B", "B", "B"]
    assert samples.end_date[0] == pd.Timestamp("2024-01-03")


def test_lstm_date_based_sequence_split_keeps_validation_later_than_train():
    end_dates = [
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-01-02"),
        pd.Timestamp("2024-01-02"),
        pd.Timestamp("2024-01-03"),
        pd.Timestamp("2024-01-03"),
        pd.Timestamp("2024-01-04"),
        pd.Timestamp("2024-01-04"),
    ]

    train_idx, valid_idx = _chronological_sequence_split_by_date(end_dates, 0.25)

    train_dates = pd.Series(end_dates).iloc[train_idx]
    valid_dates = pd.Series(end_dates).iloc[valid_idx]
    assert train_dates.max() < valid_dates.min()


def test_lstm_regressor_fits_scaler_on_train_rows_and_predicts_finite_values():
    periods = 48
    dates = pd.date_range("2024-01-01", periods=periods, freq="D")
    frame = pd.DataFrame(
        {
            "date": dates,
            "ticker": "A",
            "target_date": dates + pd.Timedelta(days=5),
            "f1": np.linspace(0.0, 1.0, periods),
            "f2": np.sin(np.arange(periods) / 3.0),
        }
    )
    target = 0.01 * np.cos(np.arange(periods) / 4.0)
    train = frame.iloc[:36].copy()
    test = frame.iloc[36:].copy()

    model = LSTMReturnRegressor(
        lookback=5,
        hidden_size=8,
        num_layers=1,
        input_projection_size=0,
        max_epochs=2,
        patience=1,
        batch_size=8,
        random_state=7,
        device="cpu",
    )
    model.fit(train, target[:36])
    pred = model.predict(test)

    assert len(pred) == len(test)
    assert np.isfinite(pred).all()
    assert model.imputer_.statistics_[0] == pytest.approx(float(np.median(train["f1"])))


def test_lstm_regressor_trains_seed_ensemble_members():
    periods = 42
    dates = pd.date_range("2024-01-01", periods=periods, freq="D")
    frame = pd.DataFrame(
        {
            "date": dates,
            "ticker": "A",
            "target_date": dates + pd.Timedelta(days=5),
            "f1": np.linspace(0.0, 1.0, periods),
            "f2": np.cos(np.arange(periods) / 4.0),
        }
    )
    target = 0.01 * np.sin(np.arange(periods) / 5.0)

    model = LSTMReturnRegressor(
        lookback=5,
        hidden_size=8,
        num_layers=1,
        input_projection_size=0,
        max_epochs=1,
        patience=1,
        batch_size=8,
        random_state=7,
        ensemble_seeds=[1, 7],
        device="cpu",
    )
    model.fit(frame.iloc[:34].copy(), target[:34])
    pred = model.predict(frame.iloc[34:].copy())

    assert len(model.models_) == 2
    assert model.training_history_["ensemble_seeds"] == [1, 7]
    assert len(pred) == len(frame.iloc[34:])
    assert np.isfinite(pred).all()


def test_lstm_regressor_supports_per_ticker_target_normalization_and_balanced_sampling():
    rows = []
    for ticker, offset, scale in [("A", 0.0, 0.01), ("B", 10.0, 0.05)]:
        dates = pd.date_range("2024-01-01", periods=36, freq="D")
        for idx, date in enumerate(dates):
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "target_date": date + pd.Timedelta(days=5),
                    "f1": offset + idx / 10.0,
                    "f2": np.sin(idx / 3.0),
                    "target": scale * np.cos(idx / 4.0),
                }
            )
    frame = pd.DataFrame(rows)

    model = LSTMReturnRegressor(
        lookback=5,
        hidden_size=8,
        num_layers=1,
        input_projection_size=0,
        max_epochs=1,
        patience=1,
        batch_size=8,
        random_state=11,
        target_normalization="per_ticker",
        balanced_ticker_sampling=True,
        device="cpu",
    )
    model.fit(frame[["date", "ticker", "target_date", "f1", "f2"]], frame["target"])
    pred = model.predict(frame.groupby("ticker", group_keys=False).tail(4)[["date", "ticker", "target_date", "f1", "f2"]])

    assert {"__global__", "A", "B"}.issubset(model.target_stats_)
    assert model.training_history_["balanced_ticker_sampling"] is True
    after = model.training_history_["balanced_ticker_sampling_info"]["after"]
    assert len(set(after.values())) == 1
    assert np.isfinite(pred).all()
