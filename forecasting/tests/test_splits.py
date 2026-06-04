import pandas as pd

from stock_forecast.splits import generate_walk_forward_splits


def test_walk_forward_splits_are_chronological_and_non_overlapping():
    df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=20)})
    splits = generate_walk_forward_splits(
        df,
        train_window=10,
        validation_window=5,
        step=5,
        mode="rolling",
    )
    assert len(splits) == 2
    for split in splits:
        assert split["train_dates"].max() < split["validation_dates"].min()
        assert set(split["train_dates"]).isdisjoint(set(split["validation_dates"]))


def test_expanding_split_keeps_first_train_date():
    df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=20)})
    splits = generate_walk_forward_splits(
        df,
        train_window=10,
        validation_window=5,
        step=5,
        mode="expanding",
    )
    assert splits[0]["train_dates"].min() == splits[1]["train_dates"].min()
