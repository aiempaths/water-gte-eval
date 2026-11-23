"""Labeling: overlap-frac rule, per-label overrides, lag buffer, priority, event_id."""

import numpy as np
import pandas as pd
import pytest

from water_gte_eval.data import TelemetryColumns, build_sample_windows, derive_intervals, label_windows


def _windows(n=48, length=12, stride=12):
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    df = pd.DataFrame({"a": np.arange(n, dtype=float)}, index=idx)
    return build_sample_windows(df, TelemetryColumns(timestamp="timestamp", sensors=("a",)),
                                length=length, stride=stride)


def _intervals(rows):
    return pd.DataFrame(rows, columns=["event_id", "start", "end", "label"])


def test_overlap_frac_rule():
    # windows are 12h, stride 12h: w0 = hours 0-11, w1 = 12-23, w2 = 24-35, w3 = 36-47
    ws = _windows()
    iv = _intervals([("ev000", "2024-01-01T12:00", "2024-01-02T00:00", "attack")])  # fully inside w1
    label_windows(ws, iv, min_overlap_frac=0.5)
    assert ws[0].label == "normal" and ws[0].event_id is None
    assert ws[1].label == "attack" and ws[1].event_id == "ev000"  # 12/12 samples covered
    assert ws[2].label == "normal"

    # 5h of a 12h window = 41.7% < 50% -> not labeled at default rule
    iv2 = _intervals([("ev000", "2024-01-01T07:00", "2024-01-01T12:00", "attack")])
    label_windows(ws, iv2, min_overlap_frac=0.5)
    assert ws[0].label == "normal"
    label_windows(ws, iv2, min_overlap_frac=0.4)
    assert ws[0].label == "attack" and ws[0].event_id == "ev000"


def test_per_label_override():
    ws = _windows()
    iv = _intervals([("ev000", "2024-01-01T07:00", "2024-01-01T12:00", "attack")])  # 5/12 = 41.7%
    label_windows(ws, iv, min_overlap_frac=0.9, label_overrides={"attack": 0.3})
    assert ws[0].label == "attack"


def test_lag_buffer_extends_interval():
    ws = _windows()
    # interval ends at hour 11 (end of w0); lag 3h extends coverage into w1 (hours 12-14)
    iv = _intervals([("ev000", "2024-01-01T00:00", "2024-01-01T12:00", "fault")])
    label_windows(ws, iv, min_overlap_frac=0.2, lag=pd.Timedelta("3h"))
    assert ws[0].label == "fault"  # 12/12
    assert ws[1].label == "fault" and ws[1].event_id == "ev000"  # 3/12 = 25% >= 20%
    label_windows(ws, iv, min_overlap_frac=0.2, lag=pd.Timedelta("0h"))
    assert ws[1].label == "normal"  # no lag -> w1 untouched


def test_priority_breaks_ties():
    ws = _windows()
    iv = _intervals([
        ("ev000", "2024-01-01T00:00", "2024-01-01T12:00", "process_change"),
        ("ev001", "2024-01-01T00:00", "2024-01-01T12:00", "attack"),
    ])
    label_windows(ws, iv, min_overlap_frac=0.5, priority=["attack", "fault", "process_change", "normal"])
    assert ws[0].label == "attack" and ws[0].event_id == "ev001"
    label_windows(ws, iv, min_overlap_frac=0.5, priority=["process_change", "attack"])
    assert ws[0].label == "process_change"


def test_derive_intervals_from_label_runs():
    idx = pd.date_range("2024-01-01", periods=48, freq="1h")
    lab = np.zeros(48, dtype=int)
    lab[10:20] = 1
    lab[30:33] = 2
    df = pd.DataFrame({"a": np.arange(48.0), "attack": lab}, index=idx)
    iv = derive_intervals(df, "attack", label_map={"0": "normal", "1": "attack", "2": "fault"})
    assert list(iv.columns) == ["event_id", "start", "end", "label"]
    assert len(iv) == 2
    assert iv.iloc[0]["label"] == "attack" and iv.iloc[0]["start"] == idx[10]
    assert iv.iloc[0]["end"] == idx[20]  # exclusive
    assert iv.iloc[1]["label"] == "fault" and iv.iloc[1]["event_id"] == "ev001"
    # feeding derived intervals into labeling gives event_ids on windows
    ws = build_sample_windows(df[["a"]], TelemetryColumns(timestamp="timestamp", sensors=("a",)),
                              length=12, stride=12)
    label_windows(ws, iv, min_overlap_frac=0.2)  # fault run covers only 3/12 samples of its window
    labeled = [w for w in ws if w.event_id is not None]
    assert {w.event_id for w in labeled} == {"ev000", "ev001"}


def test_missing_interval_columns_rejected():
    with pytest.raises(ValueError, match="event_id"):
        label_windows(_windows(24, 12, 12), pd.DataFrame({"start": [], "end": [], "label": []}))
