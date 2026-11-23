"""Feature extraction shared by the baselines.

summary_stats: per tag (sorted) [mean, std, min, max, slope] — the PLAIN
summary statistics used by SimpleKNNTrust (deliberately NOT the engine's
geometric fingerprint, which adds kurtosis and lag-1 autocorrelation).

raw_downsampled: per tag, the raw values resampled to a fixed length (deterministic
linspace index sampling) — used by the kNN-on-raw baseline.
"""

from __future__ import annotations

import numpy as np


def _sensor_array(window, tag: str) -> np.ndarray:
    return np.asarray(window.sensors[tag], dtype=np.float64)


def summary_stats(window) -> np.ndarray:
    parts: list[float] = []
    for tag in sorted(window.sensors):
        x = _sensor_array(window, tag)
        n = x.size
        if n == 0:
            parts.extend([0.0, 0.0, 0.0, 0.0, 0.0])
            continue
        mean = float(x.mean())
        std = float(x.std())
        mn = float(x.min())
        mx = float(x.max())
        slope = float(np.polyfit(np.arange(n, dtype=np.float64), x, 1)[0]) if n >= 2 else 0.0
        parts.extend([mean, std, mn, mx, slope])
    return np.asarray(parts, dtype=np.float64)


def raw_downsampled(window, d: int) -> np.ndarray:
    d = int(d)
    if d < 1:
        raise ValueError("downsample length must be >= 1")
    parts: list[float] = []
    for tag in sorted(window.sensors):
        x = _sensor_array(window, tag)
        if x.size == 0:
            parts.extend([0.0] * d)
            continue
        idx = np.linspace(0, x.size - 1, d).round().astype(int)
        parts.extend(x[idx].tolist())
    return np.asarray(parts, dtype=np.float64)
