"""Baseline 3: autoencoder reconstruction error on summary stats, implemented as a
small sklearn MLPRegressor with a bottleneck hidden layer (no PyTorch needed).
Untrust = reconstruction MSE (higher = more anomalous)."""

from __future__ import annotations

import numpy as np
from sklearn.neural_network import MLPRegressor

from .base import Baseline
from .features import summary_stats


class AutoencoderSummary(Baseline):
    name = "autoencoder"
    PARAMS = ("hidden", "max_iter")

    def __init__(self, seed: int = 42, hidden: int = 8, max_iter: int = 300):
        super().__init__(seed)
        self.hidden = max(1, int(hidden))
        self.max_iter = int(max_iter)

    def fit(self, train_windows: list, labels: list) -> None:
        X = np.array([summary_stats(w) for w in train_windows], dtype=np.float64)
        self.ae = MLPRegressor(
            hidden_layer_sizes=(self.hidden,),
            activation="relu",
            solver="adam",
            max_iter=self.max_iter,
            random_state=self.seed,
            shuffle=False,
        )
        self.ae.fit(X, X)
        self._fitted = True

    def score(self, window) -> float:
        self._check()
        x = summary_stats(window)[None, :]
        recon = self.ae.predict(x)
        return float(np.mean((recon - x) ** 2))

    def describe(self) -> dict:
        d = super().describe()
        d["params"] = {"hidden": self.hidden, "max_iter": self.max_iter}
        return d
