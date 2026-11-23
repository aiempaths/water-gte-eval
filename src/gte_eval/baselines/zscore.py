"""Baseline 1: z-score limit checks on per-tag summary stats.

Untrust score = max |z| over all (tag, stat) dimensions, using train mean/std.
"""

from __future__ import annotations

import numpy as np

from .base import Baseline
from .features import summary_stats


class ZScoreLimits(Baseline):
    name = "zscore"
    PARAMS = ("limit",)

    def __init__(self, seed: int = 42, limit: float = 3.0):
        super().__init__(seed)
        self.limit = float(limit)

    def fit(self, train_windows: list, labels: list) -> None:
        X = np.array([summary_stats(w) for w in train_windows], dtype=np.float64)
        self.mu = X.mean(axis=0)
        self.sd = X.std(axis=0)
        self.sd = np.where(self.sd < 1e-12, 1.0, self.sd)
        self._fitted = True

    def score(self, window) -> float:
        self._check()
        z = np.abs((summary_stats(window) - self.mu) / self.sd)
        return float(z.max())

    def describe(self) -> dict:
        d = super().describe()
        d["params"] = {"limit": self.limit}
        return d
