"""Baseline 4: kNN distance on RAW downsampled values (no summary stats).
Untrust = mean euclidean distance to the k nearest train windows, in a space
standardized by train stats."""

from __future__ import annotations

import numpy as np

from .base import Baseline
from .features import raw_downsampled


class KNNDistanceRaw(Baseline):
    name = "knn_raw"
    PARAMS = ("k", "downsample")

    def __init__(self, seed: int = 42, k: int = 5, downsample: int = 16):
        super().__init__(seed)
        self.k = max(1, int(k))
        self.downsample = max(1, int(downsample))

    def fit(self, train_windows: list, labels: list) -> None:
        V = np.array([raw_downsampled(w, self.downsample) for w in train_windows],
                     dtype=np.float64)
        self.mu = V.mean(axis=0)
        self.sd = V.std(axis=0)
        self.sd = np.where(self.sd < 1e-12, 1.0, self.sd)
        self.V = (V - self.mu) / self.sd
        self._fitted = True

    def score(self, window) -> float:
        self._check()
        x = (raw_downsampled(window, self.downsample) - self.mu) / self.sd
        d = np.linalg.norm(self.V - x, axis=1)
        order = np.argsort(d, kind="stable")[: self.k]
        return float(d[order].mean())

    def describe(self) -> dict:
        d = super().describe()
        d["params"] = {"k": self.k, "downsample": self.downsample}
        return d
