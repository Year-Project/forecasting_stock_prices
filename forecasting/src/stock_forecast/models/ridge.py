from __future__ import annotations

from sklearn.linear_model import Ridge

from .base import ModelSpec


def build_ridge_model(
    random_state: int = 42,
    params: dict[str, object] | None = None,
) -> ModelSpec:
    defaults: dict[str, object] = {"alpha": 1.0, "random_state": random_state}
    defaults.update(params or {})
    return ModelSpec("ridge", Ridge(**defaults), scale_features=True)
