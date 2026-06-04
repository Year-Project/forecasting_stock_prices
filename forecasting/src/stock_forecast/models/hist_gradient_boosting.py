from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingRegressor

from .base import ModelSpec


def build_hist_gradient_boosting_model(
    random_state: int = 42,
    params: dict[str, object] | None = None,
) -> ModelSpec:
    defaults: dict[str, object] = {
        "max_iter": 300,
        "learning_rate": 0.05,
        "l2_regularization": 0.01,
        "random_state": random_state,
    }
    defaults.update(params or {})
    return ModelSpec("hist_gradient_boosting", HistGradientBoostingRegressor(**defaults))


def build_sklearn_boosting_fallback(random_state: int, requested: str) -> ModelSpec:
    spec = build_hist_gradient_boosting_model(random_state=random_state)
    return ModelSpec(f"{requested}_fallback_hist_gradient_boosting", spec.estimator)
