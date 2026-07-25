"""Forecast evaluation metrics and rolling-origin backtesting."""

from typing import Callable

import numpy as np
import pandas as pd


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mape(y_true, y_pred, eps: float = 1e-8) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100)


def smape(y_true, y_pred, eps: float = 1e-8) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    denom = np.abs(y_true) + np.abs(y_pred) + eps
    return float(np.mean(2 * np.abs(y_true - y_pred) / denom) * 100)


def evaluate_all(y_true, y_pred) -> dict:
    return {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
        "sMAPE": smape(y_true, y_pred),
    }


def rolling_origin_backtest(
    df: pd.DataFrame,
    target_col: str,
    fit_predict_fn: Callable[[pd.DataFrame, pd.DataFrame], np.ndarray],
    initial_train_end: str,
    horizon_days: int = 30,
    n_folds: int = 6,
) -> pd.DataFrame:
    """
    Rolling-origin backtest: expand the training window by `horizon_days`
    each fold and evaluate on the next `horizon_days` block.

    `fit_predict_fn(train_df, test_df) -> np.ndarray` must be supplied by
    the caller and wraps whichever model (SARIMA/Prophet/LightGBM/LSTM)
    is being evaluated.
    """
    results = []
    train_end = pd.Timestamp(initial_train_end)

    for fold in range(n_folds):
        test_start = train_end
        test_end = test_start + pd.Timedelta(days=horizon_days)

        train_df = df[df.index < test_start]
        test_df = df[(df.index >= test_start) & (df.index < test_end)]

        if len(test_df) == 0:
            break

        preds = fit_predict_fn(train_df, test_df)
        scores = evaluate_all(test_df[target_col].values, preds)
        scores.update({"fold": fold, "train_end": test_start, "test_end": test_end})
        results.append(scores)

        train_end = test_end

    return pd.DataFrame(results)
