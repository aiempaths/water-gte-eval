"""Window labeling from labeled intervals.

Intervals are rows of a DataFrame with columns [event_id, start, end, label]
(half-open [start, end)). They come either from an explicit intervals file or
are derived from runs of a sample-level label column.

Labeling rule (explicit and configurable): a window gets label L from an
interval with label L when at least `min_overlap_frac` (global, or per-label via
`label_overrides`) of the window's SAMPLES fall in [start, end + lag). The lag
buffer extends each interval past its end because effects can persist after the
label ends. When several labels qualify, `priority` decides (default
attack > fault > process_change > normal). Unlabeled windows get
`default_label` and event_id None. Every labeled window carries the event_id of
its winning interval (for event-level bootstrapping).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .windows import Window

DEFAULT_PRIORITY = ["attack", "fault", "process_change", "normal"]


def derive_intervals(
    df: pd.DataFrame,
    label_col: str,
    label_map: dict | None = None,
    default_label: str = "normal",
    sentinels: list | None = None,
) -> pd.DataFrame:
    """Derive labeled intervals from contiguous runs of a sample-level label
    column. Runs equal to default_label are background and get no interval.
    sentinels: raw label values (e.g. -1, -999, "") that mean MISSING — they are
    mapped to default_label (background) and never produce intervals."""
    if label_col not in df.columns:
        raise ValueError(
            f"label column {label_col!r} not in frame; columns: {list(df.columns)}"
        )
    sent = {str(x) for x in (sentinels or [])}
    s = df[label_col]
    if label_map:
        norm = {str(k): v for k, v in label_map.items()}
        s = s.map(lambda v: norm.get(str(v), default_label if pd.isna(v) else v))
    s = s.map(lambda v: default_label if (pd.isna(v) or str(v) in sent) else v)
    s = s.fillna(default_label).astype(str)
    idx = df.index
    step = pd.Series(idx).diff().dropna().median() if len(idx) >= 2 else pd.Timedelta(0)
    vals = s.to_numpy()
    n = len(vals)
    change = np.concatenate([[True], vals[1:] != vals[:-1]])
    run_starts = np.flatnonzero(change)
    run_ends = np.concatenate([run_starts[1:], [n]])  # exclusive
    rows = []
    k = 0
    for a, b in zip(run_starts, run_ends):
        label = str(vals[a])
        if label == default_label:
            continue
        start = idx[a]
        end = idx[b] if b < n else idx[-1] + step
        rows.append({"event_id": f"ev{k:03d}", "start": start, "end": end, "label": label})
        k += 1
    return pd.DataFrame(rows, columns=["event_id", "start", "end", "label"])


def label_windows(
    windows: list[Window],
    intervals: pd.DataFrame,
    min_overlap_frac: float = 0.5,
    label_overrides: dict | None = None,
    lag: pd.Timedelta = pd.Timedelta(0),
    priority: list[str] | None = None,
    default_label: str = "normal",
) -> list[Window]:
    """Set label/event_id on every window in place (and return them)."""
    if not 0 < min_overlap_frac <= 1:
        raise ValueError("min_overlap_frac must be in (0, 1]")
    priority = list(priority or DEFAULT_PRIORITY)
    rank = {label: i for i, label in enumerate(priority)}
    overrides = {str(k): float(v) for k, v in (label_overrides or {}).items()}
    lag = pd.Timedelta(lag)
    iv = intervals.copy()
    for col in ("event_id", "start", "end", "label"):
        if col not in iv.columns:
            raise ValueError(f"intervals must have columns event_id/start/end/label; got {list(iv.columns)}")
    iv = iv.sort_values(["start", "event_id"]).reset_index(drop=True)
    iv_start = pd.to_datetime(iv["start"]).to_numpy(dtype="datetime64[ns]").view("int64")
    iv_end = (pd.to_datetime(iv["end"]) + lag).to_numpy(dtype="datetime64[ns]").view("int64")

    for w in windows:
        ts = pd.to_datetime(pd.Series(w.timestamps)).to_numpy(dtype="datetime64[ns]").view("int64")
        n = len(ts)
        best = None  # (rank, -frac, start, event_id, label)
        for j in range(len(iv)):
            lo = int(np.searchsorted(ts, iv_start[j], side="left"))
            hi = int(np.searchsorted(ts, iv_end[j], side="left"))
            frac = (hi - lo) / n if n else 0.0
            if frac <= 0:
                continue
            label = str(iv["label"].iloc[j])
            min_frac = overrides.get(label, min_overlap_frac)
            if frac >= min_frac:
                cand = (rank.get(label, len(priority)), -frac, int(iv_start[j]),
                        str(iv["event_id"].iloc[j]), label)
                if best is None or cand < best:
                    best = cand
        if best is not None:
            w.label = best[4]
            w.event_id = best[3]
            w.event_key = best[3]
        else:
            w.label = default_label
            w.event_id = None
            w.event_key = None
    return windows
