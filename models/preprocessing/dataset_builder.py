import numpy as np
import pandas as pd

from models.preprocessing.feature_generator import *
from models.preprocessing.preprocessor import *


class DatasetBuilder:
    def __init__(self, input_window: int = 100, horizon: int = 90):

        self.input_window = input_window
        self.horizon = horizon

        self.ticker_map = {}
        self.inverse_ticker_map = {}

    def fit_ticker_encoder(self, df: pd.DataFrame):
        unique_tickers = df["name"].unique()

        self.ticker_map = {
            ticker: idx for idx, ticker in enumerate(unique_tickers)
        }

        self.inverse_ticker_map = {
            idx: ticker for ticker, idx in self.ticker_map.items()
        }

    def encode_ticker(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["ticker_id"] = df["name"].map(self.ticker_map)
        return df

    def process(self, df: pd.DataFrame):
        df = df.copy()

        df = sort_by_time(df)
        df = remove_duplicates(df)
        df = fill_missing(df)

        if not self.ticker_map:
            self.fit_ticker_encoder(df)

        df = self.encode_ticker(df)

        df["log_return"] = log_return(df)

        df["return"] = simple_return(df)
        df["volatility"] = rolling_volatility(df)

        df["rsi"] = rsi(df)
        df = df.join(macd(df))
        df = df.join(bollinger_bands(df))

        df["ma_20"] = moving_average(df, 20)
        df["ema_20"] = ema(df, 20)

        df["intraday"] = intraday_return(df)
        df["overnight"] = overnight_return(df)
        df["amplitude"] = amplitude(df)

        df = add_time_features(df)

        df = df.dropna()

        return df

    def normalize(self, df: pd.DataFrame):
        df = df.copy()

        feature_cols = df.columns.drop(["log_return", "name"])

        scaler = StandardScalerTS()
        df[feature_cols] = scaler.fit_transform(df[feature_cols])

        return df

    def create_windows(self, df: pd.DataFrame):

        feature_cols = df.columns.drop(["log_return", "name"])

        return create_windows(
            df,
            feature_cols=feature_cols,
            target_col="log_return",
            input_window=self.input_window,
            horizon=self.horizon
        )

    def run(self, df: pd.DataFrame):
        df = self.process(df)

        df = self.normalize(df)

        X, y = self.create_windows(df)

        return X, y
