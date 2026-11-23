"""Per-window metrics. No point-adjusted F1 anywhere in this harness.

A window is positive iff its label equals the config's positive label; it is
flagged iff the downstream decision routes it to a human. Metrics are undefined
(None in JSON) when their denominator is zero; counts are always reported so
any convention can be recomputed from them.
"""

from __future__ import annotations


def _div(a: int, b: int) -> float | None:
    return a / b if b else None


def f1_from(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None or precision + recall == 0:
        return None
    return 2 * precision * recall / (precision + recall)


def _counts(rows: list[dict], positive: str) -> tuple[int, int, int, int]:
    tp = sum(1 for r in rows if r["label"] == positive and r["flag"])
    fp = sum(1 for r in rows if r["label"] != positive and r["flag"])
    fn = sum(1 for r in rows if r["label"] == positive and not r["flag"])
    tn = sum(1 for r in rows if r["label"] != positive and not r["flag"])
    return tp, fp, tn, fn


def window_metrics(rows: list[dict], positive: str) -> dict:
    """Aggregate metrics over individual windows. ``rows`` have label/flag keys."""
    tp, fp, tn, fn = _counts(rows, positive)
    n = len(rows)
    precision = _div(tp, tp + fp)
    recall = _div(tp, tp + fn)
    return {
        "n_windows": n,
        "positive_windows": tp + fn,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1_from(precision, recall),
        "accuracy": _div(tp + tn, n),
        "flag_rate": _div(tp + fp, n),
    }


def per_scenario(rows: list[dict], positive: str) -> dict:
    """Window-level metrics broken down by scenario id."""
    out: dict[str, dict] = {}
    scenarios = sorted({r["scenario"] for r in rows})
    for sc in scenarios:
        out[sc] = window_metrics([r for r in rows if r["scenario"] == sc], positive)
    return out
