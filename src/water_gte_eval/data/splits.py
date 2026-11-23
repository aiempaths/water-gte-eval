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
