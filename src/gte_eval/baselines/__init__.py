"""Baselines for the GTE evaluation harness.

Common interface (base.Baseline):
    fit(train_windows, labels)   # train_windows: list[gte_eval Window], labels aligned
    score(window) -> float       # single untrust score, HIGHER = less trustworthy

All baselines emit that one score, so they all share the same metrics code.
Windows are gte_eval.engine_adapter.Window objects (sensors: {tag: [float, ...]}).
"""

from .base import Baseline, BaselineError, NotFittedError
from .features import raw_downsampled, summary_stats
from .zscore import ZScoreLimits
from .iforest import IForestSummary
from .autoencoder import AutoencoderSummary
from .knn_raw import KNNDistanceRaw
from .conformal import SplitConformal
from .simple_knn_trust import SimpleKNNTrust
from .registry import make_baseline, REGISTRY

__all__ = [
    "Baseline",
    "BaselineError",
    "NotFittedError",
    "raw_downsampled",
    "summary_stats",
    "ZScoreLimits",
    "IForestSummary",
    "AutoencoderSummary",
    "KNNDistanceRaw",
    "SplitConformal",
    "SimpleKNNTrust",
    "make_baseline",
    "REGISTRY",
]
