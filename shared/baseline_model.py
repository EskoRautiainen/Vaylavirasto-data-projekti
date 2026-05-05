from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BaselineDistanceModel:
    """
    One-class baseline distance model.

    The model is fitted on good-road baseline data only. It uses robust-style
    scaled features, clips negative values to zero (upward-only deviation), and
    computes squared Mahalanobis distance from the baseline center.
    """

    mean_: np.ndarray
    inv_cov_: np.ndarray
    threshold_: float
    feature_names_: list[str]
    threshold_quantile_: float

    def _prepare(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            X_arr = X.loc[:, self.feature_names_].to_numpy(dtype=float, copy=False)
        else:
            X_arr = np.asarray(X, dtype=float)
        # Only upward deviations from baseline center contribute to badness.
        return np.clip(X_arr, a_min=0.0, a_max=None)

    def score_samples(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        Z = self._prepare(X)
        diff = Z - self.mean_
        # Squared Mahalanobis distance per row
        return np.einsum("ij,jk,ik->i", diff, self.inv_cov_, diff)

    def decision_function(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        # Larger = more normal, smaller = more anomalous (sklearn-like semantics)
        d2 = self.score_samples(X)
        return self.threshold_ - d2

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        d2 = self.score_samples(X)
        return np.where(d2 > self.threshold_, -1, 1).astype(int)
