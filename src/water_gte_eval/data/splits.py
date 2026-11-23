"""Deterministic splits: by time, and by scenario/event. Never random.

History windows all end at or before ``history_end``; eval windows start at or
after ``eval_start`` and end at or before ``eval_end``, excluding any
``holdout_scenarios``. At scoring time, ``eligible_history`` additionally
guarantees that only history windows ending strictly before the scored
window's start are visible to the engine.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .windows import Window


class SplitError(Exception):
    """Raised when a split is empty or temporally inconsistent."""


@dataclass
class Split:
    history: list[Window]
    eval: list[Window]


def temporal_split(
    windows: list[Window],
    history_end: pd.Timestamp,
    eval_start: pd.Timestamp,
    eval_end: pd.Timestamp,
    holdout_scenarios: list[str] | None = None,
) -> Split:
    history_end = pd.Timestamp(history_end)
    eval_start = pd.Timestamp(eval_start)
    eval_end = pd.Timestamp(eval_end)
    if history_end > eval_start:
        raise SplitError(
            f"split.history_end ({history_end}) must be <= split.eval_start ({eval_start}): "
            "history may only contain data from before the evaluated windows"
        )
    if eval_end <= eval_start:
        raise SplitError(f"split.eval_end ({eval_end}) must be after split.eval_start ({eval_start})")
    holdout = set(holdout_scenarios or [])
    history = [w for w in windows if w.end <= history_end]
    eval_windows = [
        w
        for w in windows
        if w.start >= eval_start and w.end <= eval_end and w.scenario not in holdout
    ]
    if not history:
        raise SplitError(
            f"no history windows: no window ends <= split.history_end ({history_end}) "
            f"within the data range; check the data and split boundaries"
        )
    if not eval_windows:
        raise SplitError(
            f"no eval windows: no window lies in [{eval_start}, {eval_end}] after removing "
            f"holdout scenarios {sorted(holdout)}; check the data and split boundaries"
        )
    key = lambda w: (w.start, w.window_id)
    return Split(history=sorted(history, key=key), eval=sorted(eval_windows, key=key))


def eligible_history(pool: list[Window], window: Window) -> list[Window]:
    """History windows visible when scoring ``window``: they must end strictly
    before the scored window starts (no overlap, no leakage)."""
    return [w for w in pool if w.end < window.start]


class LeakageError(Exception):
    """Raised by check_leakage when splits leak (time overlap or shared event)."""


@dataclass
class StageSplit:
    train: list[Window]
    val: list[Window]
    test: list[Window]
    cuts: dict  # {"val_cut": int, "test_cut": int} positions in the time-sorted list


def _straddles_event(windows: list[Window], c: int) -> bool:
    """True if an event's windows sit on both sides of the cut at index c."""
    if c <= 0 or c >= len(windows):
        return True
    a, b = windows[c - 1], windows[c]
    return a.event_id is not None and a.event_id == b.event_id


def _snap_event_cut(windows: list[Window], c: int) -> int:
    """Move the cut to the nearest position that does not split an event."""
    if not _straddles_event(windows, c):
        return c
    for d in range(1, len(windows)):
        for cand in (c - d, c + d):
            if 0 < cand < len(windows) and not _straddles_event(windows, cand):
                return cand
    raise SplitError("cannot place a split cut without splitting an event")


def _trim_overlap(earlier: list[Window], later: list[Window]) -> tuple[list[Window], int]:
    """Guard band: drop windows from the earlier stage that overlap the later
    stage in time (possible when stride < length). Returns (kept, n_dropped)."""
    if not later:
        return earlier, 0
    first_start = min(w.start for w in later)
    kept = [w for w in earlier if w.end <= first_start]
    return kept, len(earlier) - len(kept)


def assign_stages(
    windows: list[Window],
    val_frac: float = 0.15,
    test_frac: float = 0.15,
) -> StageSplit:
    """Time-ordered train/val/test split. Cuts are placed at the requested
    fractions of the time-sorted window list, snapped so an entire event stays
    in one split, then guard-banded so no window of one stage overlaps a window
    of another stage in time (boundary windows of the earlier stage are dropped
    when stride < length). Sets window.stage = train|val|test."""
    if not (0 < val_frac < 1 and 0 < test_frac < 1 and val_frac + test_frac < 1):
        raise SplitError("need 0 < val_frac, 0 < test_frac, val_frac + test_frac < 1")
    ws = sorted(windows, key=lambda w: (w.start, w.window_id))
    n = len(ws)
    if n < 3:
        raise SplitError(f"need at least 3 windows to split, got {n}")
    c_val = _snap_event_cut(ws, int(round(n * (1 - val_frac - test_frac))))
    c_test = _snap_event_cut(ws, int(round(n * (1 - test_frac))))
    if not (0 < c_val < c_test < n):
        raise SplitError(
            f"cuts not ordered after snapping: val={c_val}, test={c_test}, n={n}; "
            "adjust val_frac/test_frac"
        )
    train = ws[:c_val]
    val = ws[c_val:c_test]
    test = ws[c_test:]
    train, d1 = _trim_overlap(train, val)
    val, d2 = _trim_overlap(val, test)
    if not (train and val and test):
        raise SplitError(
            "a stage is empty after guard-band trimming; adjust val_frac/test_frac "
            "or window length/stride"
        )
    for w in train:
        w.stage = "train"
    for w in val:
        w.stage = "val"
    for w in test:
        w.stage = "test"
    return StageSplit(train=train, val=val, test=test,
                      cuts={"val_cut": c_val, "test_cut": c_test, "n": n,
                            "guard_band_dropped": {"train_val": d1, "val_test": d2}})


def _overlapping_pairs(a: list[Window], b: list[Window], cap: int = 10) -> list[tuple[Window, Window]]:
    """Half-open [start, end) time-overlap pairs between two start-sorted lists."""
    a = sorted(a, key=lambda w: w.start)
    b = sorted(b, key=lambda w: w.start)
    out = []
    j = 0
    for x in a:
        while j < len(b) and b[j].start < x.end:
            if b[j].end > x.start:
                out.append((x, b[j]))
                if len(out) >= cap:
                    return out
            j += 1
    return out


def check_leakage(train: list[Window], val: list[Window], test: list[Window]) -> dict:
    """Fail loudly (LeakageError) if any window in one stage overlaps a window in
    another stage IN TIME, or if two stages share an event_id. Returns a report
    when clean."""
    stages = {"train": train, "val": val, "test": test}
    names = list(stages)
    violations: list[str] = []
    n_overlaps = 0
    n_shared = 0
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            na, nb = names[i], names[j]
            pairs = _overlapping_pairs(stages[na], stages[nb])
            n_overlaps += len(pairs)
            for x, y in pairs:
                violations.append(
                    f"time overlap: {na} {x.window_id} [{x.start}, {x.end}) vs "
                    f"{nb} {y.window_id} [{y.start}, {y.end})"
                )
            ea = {w.event_id for w in stages[na] if w.event_id is not None}
            eb = {w.event_id for w in stages[nb] if w.event_id is not None}
            shared = sorted(ea & eb)
            n_shared += len(shared)
            for e in shared:
                violations.append(f"event_id {e!r} shared between {na} and {nb}")
    if violations:
        shown = "; ".join(violations[:20])
        more = f" (+{len(violations) - 20} more)" if len(violations) > 20 else ""
        raise LeakageError(
            f"LEAKAGE DETECTED: {len(violations)} violation(s) — {shown}{more}"
        )
    return {
        "ok": True,
        "n_time_overlaps": 0,
        "n_shared_event_ids": 0,
        "stages": {k: len(v) for k, v in stages.items()},
    }


def split_summary(split: StageSplit) -> dict:
    """Window counts per split and per label."""
    out = {"n_windows": len(split.train) + len(split.val) + len(split.test), "stages": {}}
    for name, ws in (("train", split.train), ("val", split.val), ("test", split.test)):
        labels: dict[str, int] = {}
        for w in ws:
            labels[w.label] = labels.get(w.label, 0) + 1
        out["stages"][name] = {
            "n": len(ws),
            "labels": dict(sorted(labels.items())),
            "start": min((w.start for w in ws), default=None),
            "end": max((w.end for w in ws), default=None),
        }
    return out


def format_split_summary(summary: dict) -> str:
    lines = [f"split summary: {summary['n_windows']} windows"]
    for name in ("train", "val", "test"):
        st = summary["stages"][name]
        labels = ", ".join(f"{k}={v}" for k, v in st["labels"].items()) or "-"
        lines.append(
            f"  {name:5s}: n={st['n']:6d}  labels: {labels}  "
            f"range: {st['start']} .. {st['end']}"
        )
    return "\n".join(lines)
