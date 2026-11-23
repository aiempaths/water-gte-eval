"""Baseline 5: split-conformal nonconformity on a simple classifier.

Train windows are split contiguously (time order) into a fit part and a
calibration part. A logistic regression on summary stats is fit on the fit part.
The untrust score for any window is the conformal nonconformity w.r.t. the
NORMAL label, i.e. P(positive label | window) — high when the window does not
look normal. Also exposes conformal_threshold(alpha): the split-conformal
quantile of calibration untrust scores."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from .base import Baseline
from .features import summary_stats


class SplitConformal(Baseline):
    name = "conformal"
    PARAMS = ("cal_frac", "positive_label")

    def __init__(self, seed: int = 42, cal_frac: float = 0.5, positive_label: str = "attack"):
        super().__init__(seed)
        if not 0 < cal_frac < 1:
            raise ValueError("cal_frac must be in (0, 1)")
        self.cal_frac = float(cal_frac)
        self.positive_label = str(positive_label)

    def fit(self, train_windows: list, labels: list) -> None:
        X = np.array([summary_stats(w) for w in train_windows], dtype=np.float64)
        y = np.array([1 if str(l) == self.positive_label else 0 for l in labels], dtype=int)
        n = len(X)
        n_cal = max(1, int(round(n * self.cal_frac)))
        X_fit, y_fit = X[: n - n_cal], y[: n - n_cal]
        X_cal = X[n - n_cal:]
        if np.unique(y_fit).size < 2:
            self._clf = None
            self._const = float(np.mean(y_fit))
        else:
            self._clf = LogisticRegression(max_iter=1000, random_state=self.seed)
            self._clf.fit(X_fit, y_fit)
        self._cal_scores = self._untrust(X_cal)
        self._fitted = True

    def _untrust(self, X: np.ndarray) -> np.ndarray:
        """P(positive label) = nonconformity w.r.t. the normal label."""
        if self._clf is None:
            return np.full(len(X), self._const)
        classes = list(self._clf.classes_)
        if 1 in classes:
            return self._clf.predict_proba(X)[:, classes.index(1)]
        return np.zeros(len(X))

    def score(self, window) -> float:
        self._check()
        return float(self._untrust(summary_stats(window)[None, :])[0])

    def conformal_threshold(self, alpha: float = 0.1) -> float:
        self._check()
        s = np.sort(self._cal_scores)
        k = int(np.ceil((len(s) + 1) * (1 - alpha)))
        return float(s[min(k, len(s)) - 1])

    def describe(self) -> dict:
        d = super().describe()
        d["params"] = {"cal_frac": self.cal_frac, "positive_label": self.positive_label}
        return d
