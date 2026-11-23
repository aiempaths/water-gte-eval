import numpy as np
import pandas as pd
import pytest

from water_gte_eval.data.columns import EventColumns, TelemetryColumns
from water_gte_eval.data.windows import BACKGROUND, build_windows

TCOLS = TelemetryColumns(timestamp="ts", sensors=("a", "b"))
ECOLS = EventColumns(scenario="scenario", start="start", end="end", label="label")


def _df(with_gap: bool = False) -> pd.DataFrame:
    ts = pd.date_range("2024-01-01T00:00", "2024-01-01T03:00", freq="5min", inclusive="left")
    df = pd.DataFrame({"ts": ts, "a": np.arange(len(ts), dtype=float), "b": 1.0})
    if with_gap:
        df = df[~((df.ts >= "2024-01-01T01:00") & (df.ts < "2024-01-01T02:00"))].reset_index(drop=True)
    return df


def _events(rows) -> pd.DataFrame:
    ev = pd.DataFrame(rows, columns=["scenario", "start", "end", "label"])
    ev["start"] = pd.to_datetime(ev["start"])
    ev["end"] = pd.to_datetime(ev["end"])
    return ev


def test_window_boundaries_and_default_label():
    df = _df()
    ev = _events([("s1", "2024-01-01T01:00", "2024-01-01T02:30", "anomaly")])
    ws = build_windows(df, ev, TCOLS, ECOLS, pd.Timedelta("1h"), pd.Timedelta("30min"),
                       pd.Timestamp("2024-01-01T00:00"), pd.Timestamp("2024-01-01T03:00"), "normal")
    assert [w.window_id for w in ws] == [f"w{i:05d}" for i in range(5)]
    assert ws[0].start == pd.Timestamp("2024-01-01T00:00") and ws[0].end == pd.Timestamp("2024-01-01T01:00")
    assert ws[0].label == "normal" and ws[0].scenario == BACKGROUND and ws[0].event_key is None
    assert ws[0].matrix.shape == (12, 2)
    assert ws[0].feature_vector().shape == (4,)
    labeled = [w for w in ws if w.event_key is not None]
    assert all(w.label == "anomaly" and w.scenario == "s1" for w in labeled)
    assert len(labeled) == 4  # windows overlapping 01:00-02:30


def test_max_overlap_labeling_wins():
    df = _df()
    ev = _events([
        ("s2", "2024-01-01T00:00", "2024-01-01T00:45", "other"),
        ("s1", "2024-01-01T01:00", "2024-01-01T02:30", "anomaly"),
    ])
    ws = build_windows(df, ev, TCOLS, ECOLS, pd.Timedelta("1h"), pd.Timedelta("30min"),
                       pd.Timestamp("2024-01-01T00:00"), pd.Timestamp("2024-01-01T03:00"), "normal")
    w0, w1 = ws[0], ws[1]  # [00:00,01:00) and [00:30,01:30)
    assert w0.label == "other" and w0.scenario == "s2"       # 45min overlap beats background
    assert w1.label == "anomaly" and w1.scenario == "s1"      # 30min s1 beats 15min s2


def test_empty_windows_are_skipped():
    df = _df(with_gap=True)
    ev = _events([])
    ws = build_windows(df, ev, TCOLS, ECOLS, pd.Timedelta("1h"), pd.Timedelta("30min"),
                       pd.Timestamp("2024-01-01T00:00"), pd.Timestamp("2024-01-01T03:00"), "normal")
    # [01:00,02:00) contains no rows -> skipped; 4 windows remain
    assert len(ws) == 4
    assert all(w.matrix.shape[0] > 0 for w in ws)


def test_nonpositive_window_rejected():
    with pytest.raises(ValueError):
        build_windows(_df(), _events([]), TCOLS, ECOLS, pd.Timedelta("0h"), pd.Timedelta("30min"),
                      pd.Timestamp("2024-01-01T00:00"), pd.Timestamp("2024-01-01T03:00"), "normal")
