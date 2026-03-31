import numpy as np
import pandas as pd


def log_return(df: pd.DataFrame, price_col: str = "close") -> pd.Series:
    return np.log(df[price_col]).diff()


def simple_return(df: pd.DataFrame, price_col: str = "close") -> pd.Series:
    return df[price_col].pct_change()


def rolling_volatility(df: pd.DataFrame, window: int = 14, price_col: str = "close") -> pd.Series:
    ret = log_return(df, price_col)
    return ret.rolling(window=window).std() * np.sqrt(252)


def zscore(df: pd.DataFrame, window: int = 20, price_col: str = "close") -> pd.Series:
    mean = df[price_col].rolling(window).mean()
    std = df[price_col].rolling(window).std()
    return (df[price_col] - mean) / std


def intraday_return(df: pd.DataFrame) -> pd.Series:
    return df['close'] / df['open'] - 1


def overnight_return(df: pd.DataFrame) -> pd.Series:
    return df['open'] / df['close'].shift(1) - 1


def amplitude(df: pd.DataFrame) -> pd.Series:
    return (df['high'] - df['low']) / ((df['high'] + df['low']) / 2)


def ohlc4(df: pd.DataFrame) -> pd.Series:
    return (df['open'] + df['high'] + df['low'] + df['close']) / 4.0


def median_price(df: pd.DataFrame) -> pd.Series:
    return (df['high'] + df['low']) / 2.0


def body_median(df: pd.DataFrame) -> pd.Series:
    return (df['open'] + df['close']) / 2.0


def rsi(df: pd.DataFrame, period: int = 14, price_col: str = "close") -> pd.Series:
    delta = df[price_col].diff()

    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)

    roll_up = up.ewm(alpha=1 / period, adjust=False).mean()
    roll_down = down.ewm(alpha=1 / period, adjust=False).mean()

    rs = roll_up / (roll_down + 1e-9)
    return 100 - (100 / (1 + rs))


def macd(df: pd.DataFrame, price_col: str = "close",
         fast: int = 12, slow: int = 26, signal: int = 9):

    ema_fast = df[price_col].ewm(span=fast, adjust=False).mean()
    ema_slow = df[price_col].ewm(span=slow, adjust=False).mean()

    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line

    return pd.DataFrame({
        "macd": macd_line,
        "macd_signal": signal_line,
        "macd_hist": hist
    })


def moving_average(df: pd.DataFrame, window: int = 20, price_col: str = "close") -> pd.Series:
    return df[price_col].rolling(window).mean()


def ema(df: pd.DataFrame, span: int = 20, price_col: str = "close") -> pd.Series:
    return df[price_col].ewm(span=span, adjust=False).mean()


def bollinger_bands(df: pd.DataFrame, window: int = 20, price_col: str = "close"):
    ma = df[price_col].rolling(window).mean()
    std = df[price_col].rolling(window).std()

    upper = ma + 2 * std
    lower = ma - 2 * std

    return pd.DataFrame({
        "bb_upper": upper,
        "bb_lower": lower,
        "bb_width": upper - lower
    })


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month

    return df
