"""Pure metric functions for the report generator. Hand-computable and unit-tested.

Conventions: scores are UNTRUST scores (higher = less trustworthy); windows are
escalated to humans in order of decreasing score. y = 1 means anomalous/wrong
(attack/fault), 0 means normal/legitimate. Event-level quantities group windows
by event_id; bootstrap resamples EVENTS (or singleton background windows), never
individual windows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def risk_coverage_points(scores, y, n_points: int = 50) -> list[tuple[float, float]]:
    """(coverage, error_rate) pairs as the least-trusted windows are routed to
    humans. coverage = fraction escalated; error_rate = fraction of the remaining
    (trusted) windows that are actually anomalous (false-trust rate)."""
    scores = np.asarray(scores, dtype=float)
    y = np.asarray(y, dtype=int)
    n = len(y)
    order = np.argsort(-scores, kind="stable")
    ys = y[order]
    points = []
    for c in np.linspace(0.0, 1.0, n_points):
        k = int(round(c * n))
        trusted = ys[k:]
        err = float(trusted.mean()) if trusted.size else 0.0
        points.append((float(c), err))
    return points


def false_trust_at_budget(scores, y, budget: float) -> dict:
    """At a review budget (fraction of windows routed to humans, highest untrust
    first): how many trusted windows were actually anomalous."""
    scores = np.asarray(scores, dtype=float)
    y = np.asarray(y, dtype=int)
    n = len(y)
    k = int(round(budget * n))
    order = np.argsort(-scores, kind="stable")
    trusted = order[k:]
    n_trusted = int(len(trusted))
    n_wrong = int(y[trusted].sum())
    return {
        "budget": float(budget),
        "n_escalated": k,
        "n_trusted": n_trusted,
        "n_wrong_trusted": n_wrong,
        "false_trust_rate": (n_wrong / n_trusted) if n_trusted else None,
    }


def auroc(y, scores) -> float | None:
    y = np.asarray(y, dtype=int)
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, scores))


def auprc(y, scores) -> float | None:
    y = np.asarray(y, dtype=int)
    if len(np.unique(y)) < 2:
        return None
    return float(average_precision_score(y, scores))


def event_detection(df: pd.DataFrame, min_detected_windows: int = 1) -> tuple[dict, dict]:
    """Per-event detection from rows with event_id/window_id/start/flag/true_label.
    Returns (per_event dict, aggregate). delay = first flagged window minus the
    event's first window, in windows and hours."""
    per_event: dict[str, dict] = {}
    ev_df = df[df["event_id"].astype(str) != ""]
    for eid, grp in ev_df.groupby("event_id"):
        grp = grp.sort_values("start")
        flags = grp["flag"].astype(bool).to_numpy()
        n_flag = int(flags.sum())
        detected = n_flag >= min_detected_windows
        first_pos = int(np.argmax(flags)) if flags.any() else None
        starts = pd.to_datetime(grp["start"])
        delay_h = None
        if detected:
            delay_h = float((starts.iloc[first_pos] - starts.iloc[0]).total_seconds() / 3600.0)
        per_event[str(eid)] = {
            "label": str(grp["true_label"].iloc[0]),
            "n_windows": int(len(grp)),
            "n_flagged": n_flag,
            "detected": bool(detected),
            "delay_windows": first_pos if detected else None,
            "delay_hours": delay_h,
        }
    n = len(per_event)
    det = [e["detected"] for e in per_event.values()]
    delays = [e["delay_hours"] for e in per_event.values() if e["detected"] and e["delay_hours"] is not None]
    agg = {
        "n_events": n,
        "n_detected": int(sum(det)),
        "fraction_detected": (sum(det) / n) if n else None,
        "mean_delay_hours": (float(np.mean(delays)) if delays else None),
        "median_delay_hours": (float(np.median(delays)) if delays else None),
    }
    return per_event, agg


def escalations_per_hour(flags, t_min, t_max) -> float | None:
    hours = (pd.Timestamp(t_max) - pd.Timestamp(t_min)).total_seconds() / 3600.0
    if hours <= 0:
        return None
    return float(np.asarray(flags, dtype=bool).sum()) / hours


def discrimination(df: pd.DataFrame, labels=("fault", "process_change")) -> dict:
    """Fault vs legitimate process_change: window-level AUROC on the subset with
    those labels (y = fault), plus event detection rate per label group."""
    sub = df[df["true_label"].isin(labels)]
    out: dict = {"labels": list(labels), "n_windows": int(len(sub))}
    if len(sub) and sub["true_label"].nunique() >= 2:
        y = (sub["true_label"] == labels[0]).astype(int).to_numpy()
        out["window_auroc_fault_vs_process_change"] = auroc(y, sub["score"].to_numpy())
    else:
        out["window_auroc_fault_vs_process_change"] = None
    per_label = {}
    for lab in labels:
        ev = sub[sub["true_label"] == lab]
        _, agg = event_detection(ev.assign(flag=ev["flag"].astype(bool)), 1)
        per_label[lab] = {"n_events": agg["n_events"], "fraction_detected": agg["fraction_detected"]}
    out["event_detection_by_label"] = per_label
    f = per_label.get(labels[0], {}).get("fraction_detected")
    p = per_label.get(labels[1], {}).get("fraction_detected")
    out["discrimination_gap"] = (f - p) if (f is not None and p is not None) else None
    return out


def bootstrap_over_events(df: pd.DataFrame, fn, n_resamples: int, seed: int, ci: float) -> dict:
    """Resample EVENTS (event_key column: event_id, or a singleton per background
    window) with replacement, recompute fn(sample_df) -> float each time, and
    return mean + percentile CI. None/nan values are dropped and counted."""
    df = df.copy()
    df["event_key"] = np.where(df["event_id"].astype(str) != "",
                               df["event_id"].astype(str),
                               "bg__" + df["window_id"].astype(str))
    units = sorted(df["event_key"].unique())
    unit_dfs = {u: df[df["event_key"] == u] for u in units}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(int(n_resamples)):
        idx = rng.integers(0, len(units), size=len(units))
        sample = pd.concat([unit_dfs[units[int(i)]] for i in idx])
        v = fn(sample)
        vals.append(np.nan if v is None else float(v))
    arr = np.asarray(vals, dtype=float)
    valid = arr[~np.isnan(arr)]
    alpha = (1.0 - ci) / 2.0
    if valid.size == 0:
        return {"mean": None, "ci_low": None, "ci_high": None, "n_resamples": int(n_resamples),
                "n_dropped": int(np.isnan(arr).sum())}
    return {
        "mean": float(np.mean(valid)),
        "ci_low": float(np.percentile(valid, 100 * alpha)),
        "ci_high": float(np.percentile(valid, 100 * (1 - alpha))),
        "n_resamples": int(n_resamples),
        "n_dropped": int(np.isnan(arr).sum()),
    }


def method_scores(df: pd.DataFrame, method: str) -> tuple[np.ndarray, np.ndarray]:
    """(scores, y) for a method. Engine methods: untrust = closeness + (1 - agreement).
    Baseline methods: the baseline__<name> score column."""
    if method.startswith("engine__"):
        s = df["engine_closeness"].astype(float) + (1.0 - df["engine_agreement"].astype(float))
    else:
        s = df[method].astype(float)
    y = df["y"].to_numpy(dtype=int)
    return s.to_numpy(), y


def rank_normalize_on_val(val_scores, scores) -> np.ndarray:
    """Map scores to [0, 1] via the empirical CDF of the VAL scores: the fraction
    of val scores below each score (half weight for ties). Deterministic.
    Used so the engine's combined score (and baseline scores) are on a common
    val-referenced scale — raw closeness + (1 - agreement) mixes scales."""
    v = np.sort(np.asarray(val_scores, dtype=float))
    s = np.asarray(scores, dtype=float)
    if v.size == 0:
        raise ValueError("val_scores is empty; cannot rank-normalize")
    lo = np.searchsorted(v, s, side="left")
    hi = np.searchsorted(v, s, side="right")
    return (lo + (hi - lo) / 2.0) / v.size
