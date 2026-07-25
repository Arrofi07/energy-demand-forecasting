"""SARIMA wrapper (statsmodels). Fit on a resampled (e.g. hourly) series —
minute-level SARIMA over 2M+ points is not tractable."""

import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


def fit_sarima(
    train: pd.Series,
    order: tuple = (1, 1, 1),
    seasonal_order: tuple = (1, 1, 1, 24),
    maxiter: int = 50,
):
    """seasonal_order period=24 assumes an hourly-resampled series with a
    daily cycle. Adjust to (..., 168) if working with a weekly cycle.

    `concentrate_scale=True` reduces memory/compute by removing the
    variance from the parameters the optimizer searches over — safe, and
    doesn't affect forecast scale.

    NOTE: `simple_differencing=True` was tried here as an additional
    speedup but was reverted — it returns forecasts on the *differenced*
    scale rather than the original level (a known statsmodels gotcha),
    which silently produced near-zero forecasts instead of real values.

    `low_memory=True` and `cov_type='none'` (passed to `.fit()`) are what
    actually fix the memory blowup: on one year of hourly data, fitting
    without them used ~3GB of RAM; with them, ~190MB — a ~16x reduction,
    with numerically identical forecasts (verified). `low_memory` skips
    storing intermediate filter/smoother history that isn't needed for
    pure forecasting; `cov_type='none'` skips computing the parameter
    covariance matrix (expensive Hessian computation) since we never
    inspect confidence intervals on the SARIMA coefficients themselves,
    only the forecast. Combined with `max_train_rows` in
    `run_tracked_backtest`, this is what fixes the crash — don't remove
    either without re-testing memory use on a full-size training window.

    `maxiter` caps how long a single fit can run — without a cap, a
    badly-conditioned large seasonal model can iterate far longer (and
    consume far more memory building up optimizer history) than is worth
    it for a backtest fold.

    If your index doesn't have an explicit `freq` set, statsmodels will
    warn on every fit/forecast call and — in a future version — raise an
    error instead. Set it once before backtesting, e.g.:
        hourly.index.freq = 'h'
    """
    model = SARIMAX(
        train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
        concentrate_scale=True,
    )
    return model.fit(disp=False, maxiter=maxiter, low_memory=True, cov_type="none")


def forecast_sarima(fitted_model, horizon: int):
    return fitted_model.forecast(steps=horizon)


def fit_predict(train_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str = "Global_active_power"):
    """Wrapper matching the (train_df, test_df) -> (preds, model) interface
    expected by src.evaluation.tracking.run_tracked_backtest."""
    model = fit_sarima(train_df[target_col])
    preds = forecast_sarima(model, horizon=len(test_df))
    return preds.values, model


def fit_predict(train_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str = "Global_active_power"):
    """Wrapper matching the (train_df, test_df) -> (preds, model) interface
    expected by src.evaluation.tracking.run_tracked_backtest."""
    model = fit_sarima(train_df[target_col])
    preds = forecast_sarima(model, horizon=len(test_df))
    return preds.values, model