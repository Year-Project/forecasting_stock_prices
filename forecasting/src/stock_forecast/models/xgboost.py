from __future__ import annotations

from .base import ModelSpec


def build_xgboost_model(
    random_state: int = 42,
    params: dict[str, object] | None = None,
) -> ModelSpec:
    from xgboost import XGBRegressor

    defaults: dict[str, object] = {
        "objective": "reg:squarederror",
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": 4,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "random_state": random_state,
        "n_jobs": 1,
        "verbosity": 0,
    }
    defaults.update(params or {})
    return ModelSpec("xgboost", XGBRegressor(**defaults))
