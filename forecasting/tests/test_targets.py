import numpy as np
import pandas as pd

from stock_forecast.targets import make_future_return_target


def test_future_log_return_target_per_ticker():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=4).tolist() * 2,
            "ticker": ["A"] * 4 + ["B"] * 4,
            "open": [1, 1, 1, 1, 2, 2, 2, 2],
            "high": [1, 2, 4, 8, 2, 4, 8, 16],
            "low": [1, 1, 2, 4, 2, 2, 4, 8],
            "close": [1, 2, 4, 8, 2, 4, 8, 16],
            "volume": [100] * 8,
        }
    )
    out = make_future_return_target(df, horizon=2)
    assert np.isclose(out.loc[0, "target_return_2"], np.log(4 / 1))
    assert np.isclose(out.loc[4, "target_return_2"], np.log(8 / 2))
    assert out.loc[0, "target_date"] == pd.Timestamp("2024-01-03")
    assert out.loc[4, "target_date"] == pd.Timestamp("2024-01-03")
    assert out.loc[2, "target_return_2"] != out.loc[4, "target_return_2"]
    assert out["target_return_2"].isna().sum() == 4
    assert out["target_date"].isna().sum() == 4


def test_next_open_future_log_return_target_per_ticker():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=4).tolist() * 2,
            "ticker": ["A"] * 4 + ["B"] * 4,
            "open": [10, 11, 12, 14, 20, 22, 24, 28],
            "high": [11, 12, 13, 15, 21, 23, 25, 29],
            "low": [9, 10, 11, 13, 19, 21, 23, 27],
            "close": [10, 12, 13, 15, 20, 24, 26, 30],
            "volume": [100] * 8,
        }
    )
    out = make_future_return_target(df, horizon=2, execution_timing="next_open")

    assert np.isclose(out.loc[0, "target_return_2_next_open"], np.log(14 / 11))
    assert np.isclose(out.loc[4, "target_return_2_next_open"], np.log(28 / 22))
    assert out.loc[0, "entry_date"] == pd.Timestamp("2024-01-02")
    assert out.loc[0, "entry_open"] == 11
    assert out.loc[0, "future_open"] == 14
    assert out.loc[0, "target_date"] == pd.Timestamp("2024-01-04")
    assert out["target_return_2_next_open"].isna().sum() == 6
    assert out["target_date"].isna().sum() == 6
