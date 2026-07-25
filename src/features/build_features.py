"""Feature engineering for tabular models (LightGBM) and general use.

Lag and rolling-window sizes are expressed in real time (1h, 24h, 168h) and
converted to row-offsets based on the DataFrame's actual sampling
frequency. This matters because some notebooks call this on raw minute-level
data (60 rows/hour) and others call it after resampling to hourly (1
row/hour) — a hardcoded row-offset would silently mean something different
(e.g. "shift 60" is 1 hour on minute data but 60 hours on hourly data).
"""

import numpy as np
import pandas as pd


def infer_periods_per_hour(df: pd.DataFrame) -> int:
    """Infer how many rows correspond to one hour from the median spacing
    of the datetime index. Works whether or not pandas' own freq attribute
    is set (e.g. after a manual resample)."""
    if len(df.index) < 2:
        raise ValueError("Need at least 2 rows to infer sampling frequency.")
    median_seconds = df.index.to_series().diff().dt.total_seconds().median()
    periods_per_hour = round(3600 / median_seconds)
    if periods_per_hour < 1:
        raise ValueError(
            f"Inferred sub-hourly-to-super-hourly ratio is invalid "
            f"(median spacing {median_seconds}s implies {periods_per_hour} periods/hour)."
        )
    return int(periods_per_hour)


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    return df


def add_lag_features(
    df: pd.DataFrame,
    target_col: str = "Global_active_power",
    periods_per_hour: int = None,
) -> pd.DataFrame:
    df = df.copy()
    periods_per_hour = periods_per_hour or infer_periods_per_hour(df)
    df["lag_1h"] = df[target_col].shift(1 * periods_per_hour)
    df["lag_24h"] = df[target_col].shift(24 * periods_per_hour)
    df["lag_168h"] = df[target_col].shift(168 * periods_per_hour)
    return df


def add_rolling_features(
    df: pd.DataFrame,
    target_col: str = "Global_active_power",
    periods_per_hour: int = None,
) -> pd.DataFrame:
    df = df.copy()
    periods_per_hour = periods_per_hour or infer_periods_per_hour(df)
    window = 24 * periods_per_hour
    df["rolling_mean_24h"] = df[target_col].shift(1).rolling(window).mean()
    df["rolling_std_24h"] = df[target_col].shift(1).rolling(window).std()
    return df


def build_feature_set(
    df: pd.DataFrame,
    target_col: str = "Global_active_power",
    periods_per_hour: int = None,
) -> pd.DataFrame:
    """Full feature pipeline. Drops rows with NaN introduced by lag/rolling
    windows (expected — for hourly data that's the first 168 hours / 1 week;
    for minute data it's the first 10,080 rows / 1 week).

    `periods_per_hour` is auto-inferred from the index if not given
    (e.g. 60 for minute-level data, 1 for hourly-resampled data). Pass it
    explicitly if you're using an unusual or irregular frequency where
    auto-inference might be unreliable.
    """
    periods_per_hour = periods_per_hour or infer_periods_per_hour(df)
    df = add_calendar_features(df)
    df = add_lag_features(df, target_col, periods_per_hour=periods_per_hour)
    df = add_rolling_features(df, target_col, periods_per_hour=periods_per_hour)
    return df.dropna()
