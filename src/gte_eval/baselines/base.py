"""Common baseline interface: fit(train_windows, labels); score(window) -> float
untrust score (higher = less trustworthy)."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaselineError(Exception):
    pass


class NotFittedError(BaselineError):
    pass


class Baseline(ABC):
    name: str = "abstract"
    PARAMS: tuple[str, ...] = ()

    def __init__(self, seed: int = 42):
        self.seed = int(seed)
        self._fitted = False

    @abstractmethod
    def fit(self, train_windows: list, labels: list) -> None:
        """Fit on train windows; labels are the aligned window labels."""

    @abstractmethod
    def score(self, window) -> float:
        """Untrust score for one window; higher = less trustworthy."""

    def score_windows(self, windows: list) -> list[float]:
        self._check()
        return [float(self.score(w)) for w in windows]

    def _check(self) -> None:
        if not self._fitted:
            raise NotFittedError(f"baseline {self.name!r} is not fitted; call fit() first")

    def describe(self) -> dict:
        return {"name": self.name, "seed": self.seed}
