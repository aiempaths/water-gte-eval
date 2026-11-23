"""Splits + leakage checker: must PASS on a proper split and FAIL loudly on a
deliberately leaky one (time overlap or shared event_id across stages)."""

import numpy as np
import pandas as pd
import pytest

from water_gte_eval.data import (
    LeakageError,
    TelemetryColumns,
    assign_stages,
    build_sample_windows,
    check_leakage,
    format_split_summary,
    label_windows,
    split_summary,
)


def _make_windows(n=240, length=12, stride=6, with_event=True):
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    df = pd.DataFrame({"a": np.arange(n, dtype=float)}, index=idx)
    ws = build_sample_windows(df, TelemetryColumns(timestamp="timestamp", sensors=("a",)),
                              length=length, stride=stride)
    if with_event:
        # one attack event around the middle of the timeline
        iv = pd.DataFrame([{
            "event_id": "ev000",
            "start": idx[100], "end": idx[130], "label": "attack",
        }])
        label_windows(ws, iv, min_overlap_frac=0.5)
    return ws


def test_assign_stages_time_ordered_and_counts():
    ws = _make_windows()
    split = assign_stages(ws, val_frac=0.2, test_frac=0.2)
    n = len(ws)
    dropped = sum(split.cuts["guard_band_dropped"].values())
    assert len(split.train) + len(split.val) + len(split.test) == n - dropped
    # time-ordered: every train window starts before every val window, etc.
    assert max(w.start for w in split.train) <= min(w.start for w in split.val)
    assert max(w.start for w in split.val) <= min(w.start for w in split.test)
    assert all(w.stage == "train" for w in split.train)
    assert all(w.stage == "test" for w in split.test)
    # roughly the requested fractions
    assert abs(len(split.test) / n - 0.2) < 0.1


def test_event_stays_in_one_split():
    ws = _make_windows(with_event=True)
    split = assign_stages(ws, val_frac=0.5, test_frac=0.3)  # raw cuts likely inside/near the event
    ev_windows = [w for w in ws if w.event_id == "ev000"]
    stages = {w.stage for w in ev_windows}
    assert len(stages) == 1, f"event ev000 split across stages: {stages}"
    check_leakage(split.train, split.val, split.test)  # must pass


def test_no_cross_stage_time_overlap_even_with_stride_lt_length():
    ws = _make_windows(length=12, stride=6)  # windows overlap each other in time
    split = assign_stages(ws, val_frac=0.2, test_frac=0.2)
    for a, b in ((split.train, split.val), (split.val, split.test), (split.train, split.test)):
        assert min(w.start for w in b) >= max(w.end for w in a), "stages overlap in time"


def test_check_leakage_passes_on_good_split():
    ws = _make_windows()
    split = assign_stages(ws, val_frac=0.15, test_frac=0.15)
    report = check_leakage(split.train, split.val, split.test)
    assert report["ok"] is True
    assert report["stages"] == {"train": len(split.train), "val": len(split.val), "test": len(split.test)}


def test_check_leakage_fails_on_time_overlap():
    ws = _make_windows(with_event=False)
    split = assign_stages(ws, val_frac=0.2, test_frac=0.2)
    # deliberately leaky: move the first test window back into train's time range
    leaky_test = list(split.test)
    leaky_test[0] = ws[0]  # ws[0] is a train window; it overlaps train windows in time
    with pytest.raises(LeakageError, match="time overlap"):
        check_leakage(split.train, split.val, leaky_test)


def test_check_leakage_fails_on_shared_event_id():
    ws = _make_windows(with_event=True)
    split = assign_stages(ws, val_frac=0.2, test_frac=0.2)
    ev_window = next(w for w in ws if w.event_id == "ev000")
    src = ev_window.stage
    dst = "test" if src != "test" else "train"
    stages = {"train": list(split.train), "val": list(split.val), "test": list(split.test)}
    stages[src].remove(ev_window)
    stages[dst].append(ev_window)  # deliberately leaky: event window in two stages' event set
    with pytest.raises(LeakageError, match="event_id"):
        check_leakage(stages["train"], stages["val"], stages["test"])


def test_leaky_raw_fraction_split_fails_checker():
    # a naive fraction split that ignores event boundaries and window overlap
    ws = _make_windows(with_event=True, length=12, stride=6)
    n = len(ws)
    naive_train, naive_val, naive_test = ws[: n // 2], ws[n // 2 : 3 * n // 4], ws[3 * n // 4 :]
    for w in naive_train:
        w.stage = "train"
    for w in naive_val:
        w.stage = "val"
    for w in naive_test:
        w.stage = "test"
    with pytest.raises(LeakageError):
        check_leakage(naive_train, naive_val, naive_test)


def test_split_summary_and_format():
    ws = _make_windows()
    split = assign_stages(ws, val_frac=0.2, test_frac=0.2)
    summary = split_summary(split)
    assert summary["n_windows"] == len(split.train) + len(split.val) + len(split.test)
    for stage in ("train", "val", "test"):
        st = summary["stages"][stage]
        assert st["n"] > 0
        assert sum(st["labels"].values()) == st["n"]
    text = format_split_summary(summary)
    assert "split summary" in text and "train" in text and "attack" in text or "normal" in text
    assert "labels:" in text
