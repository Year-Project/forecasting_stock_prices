import numpy as np
import pandas as pd


def sort_by_time(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_index()


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return df[~df.index.duplicated(keep='first')]


def fill_missing(df: pd.DataFrame, method: str = "ffill") -> pd.DataFrame:
    if method == "ffill":
        return df.ffill()
    elif method == "bfill":
        return df.bfill()
    else:
        return df.fillna(0)


class StandardScalerTS:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, df: pd.DataFrame):
        self.mean = df.mean()
        self.std = df.std() + 1e-9

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return (df - self.mean) / self.std

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.fit(df)
        return self.transform(df)


def train_val_test_split(df: pd.DataFrame, train_ratio=0.7, val_ratio=0.15):
    n = len(df)

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]

    return train, val, test


def create_windows(df: pd.DataFrame,
                   feature_cols: list,
                   target_col: str,
                   input_window: int = 100,
                   horizon: int = 90):
    X = []
    y = []

    values = df[feature_cols].values
    target = df[target_col].values

    for i in range(input_window, len(df) - horizon):
        X.append(values[i - input_window:i])
        y.append(target[i:i + horizon])

    return np.array(X), np.array(y)
