from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin


@dataclass
class ModelSpec:
    name: str
    estimator: Any
    scale_features: bool = False
    use_pipeline: bool = True
    input_mode: str = "features"


class BaseReturnModel(BaseEstimator, RegressorMixin):
    fallback_: float = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaseReturnModel":
        self.fallback_ = self._target_mean(y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    def _target_mean(y: pd.Series) -> float:
        values = np.asarray(y, dtype=float)
        values = values[np.isfinite(values)]
        if len(values) == 0:
            return 0.0
        return float(values.mean())

    def _fallback_prediction(self, n_rows: int) -> np.ndarray:
        return np.full(n_rows, self.fallback_, dtype=float)
