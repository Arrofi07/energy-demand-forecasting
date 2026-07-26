"""Prophet wrapper. Expects a df with columns ['ds', 'y']."""

import pandas as pd
from prophet import Prophet


def fit_prophet(train: pd.DataFrame, daily: bool = True, weekly: bool = True) -> Prophet:
    model = Prophet(
        daily_seasonality=daily,
        weekly_seasonality=weekly,
        yearly_seasonality=True,
    )
    model.fit(train)
    return model


def forecast_prophet(model: Prophet, future_df: pd.DataFrame) -> pd.DataFrame:
    """future_df needs a 'ds' column covering the forecast horizon."""
    return model.predict(future_df)[["ds", "yhat", "yhat_lower", "yhat_upper"]]


def fit_predict(train_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str = "Global_active_power"):
    """Wrapper matching the (train_df, test_df) -> (preds, model) interface
    expected by src.evaluation.tracking.run_tracked_backtest. Converts the
    datetime-indexed train/test frames to Prophet's required ds/y format."""
    train_prophet = train_df[[target_col]].reset_index()
    train_prophet.columns = ["ds", "y"]

    future = test_df[[target_col]].reset_index()
    future.columns = ["ds", "y"]  # y dropped below, kept only for column reset consistency
    future = future[["ds"]]

    model = fit_prophet(train_prophet)
    forecast = forecast_prophet(model, future)
    return forecast["yhat"].values, model
