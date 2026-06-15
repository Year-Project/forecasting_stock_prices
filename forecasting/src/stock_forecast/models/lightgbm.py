from __future__ import annotations

from .base import ModelSpec


def build_lightgbm_model(
    random_state: int = 42,
    params: dict[str, object] | None = None,
) -> ModelSpec:
    from lightgbm import LGBMRegressor

    defaults: dict[str, object] = {
        "objective": "regression",
        "n_estimators": 500,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "random_state": random_state,
        "n_jobs": 1,
        "verbosity": -1,
        "force_col_wise": True,
    }
    defaults.update(params or {})
    return ModelSpec("lightgbm", LGBMRegressor(**defaults))
