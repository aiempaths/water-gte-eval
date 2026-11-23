"""Baseline 2: IsolationForest on summary stats. Untrust = -decision_function
(higher = more anomalous)."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest

from .base import Baseline
from .features import summary_stats


class IForestSummary(Baseline):
    name = "iforest"
    PARAMS = ("n_estimators", "max_samples", "contamination")

    def __init__(self, seed: int = 42, n_estimators: int = 100, max_samples: int = 256,
                 contamination: float = 0.05):
        super().__init__(seed)
        self.n_estimators = int(n_estimators)
        self.max_samples = max_samples
        self.contamination = float(contamination)

    def fit(self, train_windows: list, labels: list) -> None:
        X = np.array([summary_stats(w) for w in train_windows], dtype=np.float64)
        self.clf = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            contamination=self.contamination,
            random_state=self.seed,
        )
        self.clf.fit(X)
        self._fitted = True

    def score(self, window) -> float:
        self._check()
        return float(-self.clf.decision_function(summary_stats(window)[None, :])[0])

    def describe(self) -> dict:
        d = super().describe()
        d["params"] = {"n_estimators": self.n_estimators, "max_samples": self.max_samples,
                       "contamination": self.contamination}
        return d
