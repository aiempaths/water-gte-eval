"""Baseline registry: config dict -> Baseline instance."""

from __future__ import annotations

from .autoencoder import AutoencoderSummary
from .base import Baseline, BaselineError
from .conformal import SplitConformal
from .iforest import IForestSummary
from .knn_raw import KNNDistanceRaw
from .simple_knn_trust import SimpleKNNTrust
from .zscore import ZScoreLimits

REGISTRY: dict[str, type[Baseline]] = {
    "zscore": ZScoreLimits,
    "iforest": IForestSummary,
    "autoencoder": AutoencoderSummary,
    "knn_raw": KNNDistanceRaw,
    "conformal": SplitConformal,
    "simple_knn_trust": SimpleKNNTrust,
}


def make_baseline(cfg: dict) -> Baseline:
    name = cfg.get("name")
    cls = REGISTRY.get(name)
    if cls is None:
        raise BaselineError(
            f"Unknown baseline {name!r}. Registered: {sorted(REGISTRY)}"
        )
    params = dict(cfg.get("params") or {})
    unknown = sorted(set(params) - set(cls.PARAMS))
    if unknown:
        raise BaselineError(f"Unknown {name} param(s) {unknown}; allowed: {sorted(cls.PARAMS)}")
    return cls(seed=int(cfg.get("seed", 42)), **params)
