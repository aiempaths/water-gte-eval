"""Window-level robustness perturbations (seeded, deterministic).

Applied to prepared Window objects after loading, before scoring, so the whole
pipeline (history eligibility, modes, baselines) stays identical and only the
telemetry seen by the scorers changes. `scope` selects which windows are
perturbed: "scored" (val+test only; train history stays clean, isolating scoring
degradation) or "all" (the whole stream is degraded).

- sensor_dropout: each tag of a perturbed window is independently "dropped" with
  probability `prob`; the dropped column is replaced by a constant (its first
  sample, `fill: hold_first` — a dead sensor reporting a stuck value — or 0.0),
  keeping the sensor dimension so history fingerprints stay comparable.
- clock_skew: each perturbed window's samples are rolled by `shift_samples`
  (same shift for all tags), simulating a clock offset so the window nominally
  covering [start, end) actually holds data shifted by that many samples.

The RNG is drawn in (start, window_id) order over the selected windows only, so
results are deterministic for a fixed seed and input window list.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .data import Window

PERTURBATIONS = ("sensor_dropout", "clock_skew")


class PerturbConfigError(Exception):
    pass


def apply_perturbation(windows: list[Window], perturb: dict, seed: int) -> list[Window]:
    if not isinstance(perturb, dict) or perturb.get("name") not in PERTURBATIONS:
        raise PerturbConfigError(
            f"perturb.name must be one of {PERTURBATIONS}, got {perturb!r}"
        )
    name = perturb["name"]
    params = perturb.get("params") or {}
    scope = perturb.get("scope", "scored")
    if scope not in ("scored", "all"):
        raise PerturbConfigError(f"perturb.scope must be 'scored' or 'all', got {scope!r}")

    order = sorted(range(len(windows)), key=lambda i: (windows[i].start, windows[i].window_id))
    selected = [i for i in order if scope == "all" or windows[i].stage in ("val", "test")]
    rng = np.random.default_rng(seed)
    out = list(windows)
    for i in selected:
        w = out[i]
        m = np.array(w.matrix, dtype=np.float64, copy=True)
        if name == "sensor_dropout":
            prob = float(params.get("prob", 0.1))
            if not 0.0 <= prob <= 1.0:
                raise PerturbConfigError("sensor_dropout prob must be in [0, 1]")
            fill = params.get("fill", "hold_first")
            if fill not in ("hold_first", "zero"):
                raise PerturbConfigError("sensor_dropout fill must be 'hold_first' or 'zero'")
            for j in range(m.shape[1]):
                if rng.random() < prob:
                    m[:, j] = m[0, j] if fill == "hold_first" else 0.0
        elif name == "clock_skew":
            k = int(params.get("shift_samples", 1))
            if k == 0:
                raise PerturbConfigError("clock_skew shift_samples must be non-zero")
            m = np.roll(m, k, axis=0)
        out[i] = replace(w, matrix=m)
    return out
