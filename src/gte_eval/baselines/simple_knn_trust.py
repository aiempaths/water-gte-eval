"""Baseline 6: SimpleKNNTrust — the SAME two-score design as the engine
(closeness + agreement) but on PLAIN summary statistics (mean, std, min, max,
slope) instead of the engine's geometric fingerprint (which adds kurtosis and
lag-1 autocorrelation). Exists to test whether the geometric fingerprint adds
anything beyond summary stats.

Untrust score = closeness + (1 - agreement), where closeness = mean distance to
the k nearest train windows in summary-stat space standardized by train stats,
and agreement = fraction of those neighbors sharing the majority outcome label.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from .base import Baseline
from .features import summary_stats


class SimpleKNNTrust(Baseline):
    name = "simple_knn_trust"
    PARAMS = ("k",)

    def __init__(self, seed: int = 42, k: int = 5):
        super().__init__(seed)
        self.k = max(1, int(k))

    def fit(self, train_windows: list, labels: list) -> None:
        X = np.array([summary_stats(w) for w in train_windows], dtype=np.float64)
        self.labels = [str(l) for l in labels]
        self.mu = X.mean(axis=0)
        self.sd = X.std(axis=0)
        self.sd = np.where(self.sd < 1e-12, 1.0, self.sd)
        self.X = (X - self.mu) / self.sd
        self._fitted = True

    def score_parts(self, window) -> tuple[float, float]:
        """(closeness, agreement) — the engine's two scores on summary stats."""
        self._check()
        x = (summary_stats(window) - self.mu) / self.sd
        d = np.linalg.norm(self.X - x, axis=1)
        order = sorted(range(len(d)), key=lambda i: (float(d[i]), i))[: self.k]
        closeness = float(d[order].mean())
        neigh_labels = [self.labels[i] for i in order]
        majority, count = sorted(Counter(neigh_labels).items(), key=lambda kv: (-kv[1], kv[0]))[0]
        agreement = float(count) / len(order)
        return closeness, agreement

    def score(self, window) -> float:
        closeness, agreement = self.score_parts(window)
        return float(closeness + (1.0 - agreement))

    def describe(self) -> dict:
        d = super().describe()
        d["params"] = {"k": self.k}
        return d
