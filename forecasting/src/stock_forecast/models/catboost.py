from __future__ import annotations

from .base import ModelSpec


def build_catboost_model(
    random_state: int = 42,
    params: dict[str, object] | None = None,
) -> ModelSpec:
    from catboost import CatBoostRegressor

    defaults: dict[str, object] = {
        "loss_function": "RMSE",
        "random_seed": random_state,
        "verbose": False,
        "allow_writing_files": False,
        "thread_count": 1,
        "iterations": 500,
        "learning_rate": 0.05,
        "depth": 6,
    }
    defaults.update(params or {})
    return ModelSpec("catboost", CatBoostRegressor(**defaults))
