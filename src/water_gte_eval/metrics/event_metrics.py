"""Event-level metrics.

Eval windows that belong to an annotated event (event_key not None) are
grouped by that event. An event counts as detected when at least
``min_detected_windows`` of its windows are flagged. This is the event-level
reporting the harness uses instead of point-adjusted F1. Windows outside any
event (background) contribute only to window-level metrics.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..data.windows import Window
from .window_metrics import _div, f1_from


@dataclass
class EventRecord:
    event_key: str
    scenario: str
    label: str
    n_windows: int
    n_flagged: int
    detected: bool


def build_event_records(
    windows: list[Window],
    flags: dict[str, bool],
    positive: str,
    min_detected_windows: int,
) -> list[EventRecord]:
    groups: dict[str, list[Window]] = {}
    for w in windows:
        if w.event_key is None:
            continue
        groups.setdefault(w.event_key, []).append(w)
    records: list[EventRecord] = []
    for key in sorted(groups):
        ws = groups[key]
        n_flagged = sum(1 for w in ws if flags.get(w.window_id, False))
        records.append(
            EventRecord(
                event_key=key,
                scenario=ws[0].scenario,
                label=ws[0].label,
                n_windows=len(ws),
                n_flagged=n_flagged,
                detected=n_flagged >= min_detected_windows,
            )
        )
    return records


def event_metrics_from_records(records: list[EventRecord], positive: str) -> dict:
    """Event-level confusion and rates from event records (used by bootstrap too)."""
    pos = [r for r in records if r.label == positive]
    neg = [r for r in records if r.label != positive]
    tp = sum(1 for r in pos if r.detected)
    fn = len(pos) - tp
    fp = sum(1 for r in neg if r.detected)
    tn = len(neg) - fp
    precision = _div(tp, tp + fp)
    recall = _div(tp, tp + fn)
    return {
        "n_events": len(records),
        "n_positive_events": len(pos),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1_from(precision, recall),
    }


def event_metrics(
    records: list[EventRecord], positive: str, eval_days: float
) -> dict:
    m = event_metrics_from_records(records, positive)
    m["false_events_per_day"] = (m["fp"] / eval_days) if eval_days > 0 else None
    return m
