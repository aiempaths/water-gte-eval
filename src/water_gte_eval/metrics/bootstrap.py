"""Bootstrap confidence intervals over events or scenarios — never over windows.

Resampling units are whole events (``unit: events``) or whole scenarios
(``unit: scenarios``, which resamples all events of a scenario together).
Seeded from the config, so re-running gives identical intervals. Resamples in
which a metric is undefined (zero denominator) are dropped from that metric's
interval and counted in ``n_dropped``.
"""

from __future__ import annotations

import numpy as np

from .event_metrics import EventRecord, event_metrics_from_records

_METRICS = ("precision", "recall", "f1")


def bootstrap_event_metrics(
    records: list[EventRecord],
    unit: str,
    n_resamples: int,
    ci: float,
    seed: int,
    positive: str,
) -> dict:
    if unit == "events":
        units = sorted({r.event_key for r in records})
    elif unit == "scenarios":
        units = sorted({r.scenario for r in records})
    else:
        raise ValueError(f"bootstrap unit must be 'events' or 'scenarios', got {unit!r}")
    if not units:
        raise ValueError("no event records to bootstrap over")
    unit_records: dict[str, list[EventRecord]] = {u: [] for u in units}
    for r in records:
        key = r.event_key if unit == "events" else r.scenario
        unit_records[key].append(r)

    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {m: [] for m in _METRICS}
    n_dropped: dict[str, int] = {m: 0 for m in _METRICS}
    for _ in range(int(n_resamples)):
        idx = rng.integers(0, len(units), size=len(units))
        sample = [r for i in idx for r in unit_records[units[int(i)]]]
        m = event_metrics_from_records(sample, positive)
        for name in _METRICS:
            v = m[name]
            if v is None:
                n_dropped[name] += 1
                draws[name].append(float("nan"))
            else:
                draws[name].append(float(v))

    alpha = (1.0 - float(ci)) / 2.0
    out: dict[str, dict] = {}
    for name in _METRICS:
        arr = np.array(draws[name], dtype=np.float64)
        valid = arr[~np.isnan(arr)]
        if valid.size == 0:
            out[name] = {"mean": None, "ci_low": None, "ci_high": None,
                         "n_valid": 0, "n_dropped": n_dropped[name]}
        else:
            out[name] = {
                "mean": float(np.mean(valid)),
                "ci_low": float(np.percentile(valid, 100 * alpha)),
                "ci_high": float(np.percentile(valid, 100 * (1 - alpha))),
                "n_valid": int(valid.size),
                "n_dropped": n_dropped[name],
            }
    return {"unit": unit, "n_resamples": int(n_resamples), "ci": float(ci), "metrics": out}
