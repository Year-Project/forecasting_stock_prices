from __future__ import annotations

import numpy as np
import pandas as pd

from .base import BaseReturnModel, ModelSpec


class NaivePersistenceReturnModel(BaseReturnModel):
    column = "ret_1"

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.column not in X.columns:
            return self._fallback_prediction(len(X))
        pred = X[self.column].to_numpy(dtype=float)
        return np.nan_to_num(pred, nan=self.fallback_)


def build_naive_persistence_model(params: dict[str, object] | None = None) -> ModelSpec:
    return ModelSpec("naive_persistence", NaivePersistenceReturnModel())
