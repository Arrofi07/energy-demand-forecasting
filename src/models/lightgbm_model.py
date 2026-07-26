"""LightGBM wrapper for the feature-based tabular approach."""

import lightgbm as lgb
import pandas as pd

DEFAULT_FEATURES = [
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "lag_1h",
    "lag_24h",
    "lag_168h",
    "rolling_mean_24h",
    "rolling_std_24h",
]


def fit_lightgbm(
    train_df: pd.DataFrame,
    target_col: str = "Global_active_power",
    features: list = None,
    params: dict = None,
) -> lgb.Booster:
    features = features or DEFAULT_FEATURES
    params = params or {
        "objective": "regression",
        "metric": "mae",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "verbose": -1,
    }
    train_set = lgb.Dataset(train_df[features], label=train_df[target_col])
    return lgb.train(params, train_set, num_boost_round=300)


def predict_lightgbm(model: lgb.Booster, test_df: pd.DataFrame, features: list = None):
    features = features or DEFAULT_FEATURES
    return model.predict(test_df[features])


def fit_predict(train_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str = "Global_active_power"):
    """Wrapper matching the (train_df, test_df) -> (preds, model) interface
    expected by src.evaluation.tracking.run_tracked_backtest. Assumes
    train_df/test_df already have engineered features (see
    src.features.build_features.build_feature_set)."""
    model = fit_lightgbm(train_df, target_col=target_col)
    preds = predict_lightgbm(model, test_df)
    return preds, model
