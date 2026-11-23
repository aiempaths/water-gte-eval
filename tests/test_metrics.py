import numpy as np
import pandas as pd

from water_gte_eval.data.windows import Window
from water_gte_eval.metrics import (
    build_event_records,
    event_metrics,
    f1_from,
    per_scenario,
    window_metrics,
)


def test_window_metrics_hand_computed():
    rows = [
        {"label": "anomaly", "flag": True, "scenario": "a"},
        {"label": "anomaly", "flag": False, "scenario": "a"},
        {"label": "normal", "flag": True, "scenario": "b"},
        {"label": "normal", "flag": False, "scenario": "b"},
    ]
    m = window_metrics(rows, "anomaly")
    assert (m["tp"], m["fp"], m["tn"], m["fn"]) == (1, 1, 1, 1)
    assert m["precision"] == 0.5 and m["recall"] == 0.5 and m["f1"] == 0.5
    assert m["accuracy"] == 0.5 and m["flag_rate"] == 0.5 and m["n_windows"] == 4


def test_window_metrics_undefined_denominators_are_none():
    rows = [{"label": "normal", "flag": False, "scenario": "a"}]
    m = window_metrics(rows, "anomaly")
    assert m["precision"] is None and m["recall"] is None and m["f1"] is None
    assert m["accuracy"] == 1.0


def test_f1_from_edges():
    assert f1_from(None, 0.5) is None
    assert f1_from(0.0, 0.0) is None
    assert f1_from(1.0, 1.0) == 1.0


def test_per_scenario_breakdown():
    rows = [
        {"label": "anomaly", "flag": True, "scenario": "a"},
        {"label": "normal", "flag": False, "scenario": "b"},
    ]
    ps = per_scenario(rows, "anomaly")
    assert set(ps) == {"a", "b"}
    assert ps["a"]["recall"] == 1.0 and ps["b"]["precision"] is None


def _win(i, event_key, label, scenario):
    return Window(
        window_id=f"w{i:05d}",
        start=pd.Timestamp("2024-06-01T00:00") + pd.Timedelta(hours=i),
        end=pd.Timestamp("2024-06-01T00:00") + pd.Timedelta(hours=i + 1),
        matrix=np.zeros((2, 1)),
        sensors=("a",),
        label=label,
        scenario=scenario,
        event_key=event_key,
    )


def test_event_records_and_metrics():
    windows = [
        _win(0, "evA|0|1", "anomaly", "s1"),
        _win(1, "evA|0|1", "anomaly", "s1"),
        _win(2, "evA|0|1", "anomaly", "s1"),
        _win(3, "evB|0|1", "normal", "s2"),
        _win(4, "evB|0|1", "normal", "s2"),
        _win(5, None, "normal", "background"),
    ]
    flags = {"w00000": True, "w00001": True, "w00002": False, "w00003": True, "w00004": False, "w00005": True}
    recs = build_event_records(windows, flags, "anomaly", min_detected_windows=2)
    assert len(recs) == 2  # background windows are not events
    a = next(r for r in recs if r.event_key.startswith("evA"))
    assert a.n_windows == 3 and a.n_flagged == 2 and a.detected
    m = event_metrics(recs, "anomaly", eval_days=2.0)
    assert m["n_events"] == 2 and m["n_positive_events"] == 1
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (1, 0, 0, 1)
    assert m["precision"] == 1.0 and m["recall"] == 1.0 and m["f1"] == 1.0
    assert m["false_events_per_day"] == 0.0


def test_min_detected_windows_threshold():
    windows = [_win(0, "evA", "anomaly", "s1"), _win(1, "evA", "anomaly", "s1")]
    flags = {"w00000": True, "w00001": True}
    recs2 = build_event_records(windows, flags, "anomaly", 2)
    recs3 = build_event_records(windows, flags, "anomaly", 3)
    assert recs2[0].detected and not recs3[0].detected
    m = event_metrics(recs3, "anomaly", eval_days=1.0)
    assert m["recall"] == 0.0 and m["fn"] == 1
