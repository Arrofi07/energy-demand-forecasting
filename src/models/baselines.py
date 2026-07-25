"""Baseline forecasting models: naive and seasonal naive."""

import numpy as np
import pandas as pd


def naive_forecast(train: pd.Series, horizon: int) -> np.ndarray:
    """Tomorrow = today: repeat the last observed value."""
    return np.full(horizon, train.iloc[-1])


def seasonal_naive_forecast(
    train: pd.Series, horizon: int, season_length: int = 60 * 24 * 7
) -> np.ndarray:
    """Same value as one season ago (default: same hour, one week back)."""
    if len(train) < season_length:
        return naive_forecast(train, horizon)
    last_season = train.iloc[-season_length:]
    reps = int(np.ceil(horizon / season_length))
    return np.tile(last_season.values, reps)[:horizon]
