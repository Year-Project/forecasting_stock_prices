from __future__ import annotations

from typing import Any

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .base import ModelSpec
from .hist_gradient_boosting import (
    build_hist_gradient_boosting_model,
    build_sklearn_boosting_fallback,
)
from .momentum import build_momentum_model
from .naive_persistence import build_naive_persistence_model
from .ridge import build_ridge_model


def build_model(
    model_type: str,
    random_state: int = 42,
    params: dict[str, Any] | None = None,
) -> ModelSpec:
    model_type = model_type.lower()
    model_params = dict(params or {})

    if model_type in {"naive_persistence", "naive_last", "persistence"}:
        return build_naive_persistence_model(model_params)
    if model_type == "momentum":
        return build_momentum_model(model_params)
    if model_type == "ridge":
        return build_ridge_model(random_state=random_state, params=model_params)
    if model_type in {"hist_gradient_boosting", "hgb"}:
        return build_hist_gradient_boosting_model(random_state=random_state, params=model_params)
    if model_type == "catboost":
        try:
            from .catboost import build_catboost_model

            return build_catboost_model(random_state=random_state, params=model_params)
        except ImportError:
            return build_sklearn_boosting_fallback(random_state, requested=model_type)
    if model_type == "xgboost":
        try:
            from .xgboost import build_xgboost_model

            return build_xgboost_model(random_state=random_state, params=model_params)
        except ImportError:
            return build_sklearn_boosting_fallback(random_state, requested=model_type)
    if model_type == "lightgbm":
        try:
            from .lightgbm import build_lightgbm_model

            return build_lightgbm_model(random_state=random_state, params=model_params)
        except ImportError:
            return build_sklearn_boosting_fallback(random_state, requested=model_type)
    if model_type == "lstm":
        from .lstm import build_lstm_model

        return build_lstm_model(random_state=random_state, params=model_params)
    raise ValueError(f"Unsupported model type: {model_type}")


def make_pipeline(spec: ModelSpec) -> Pipeline:
    if not spec.use_pipeline:
        return spec.estimator
    steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if spec.scale_features:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", spec.estimator))
    pipeline = Pipeline(steps)
    try:
        pipeline.set_output(transform="pandas")
    except ValueError:
        pass
    return pipeline
