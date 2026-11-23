"""Hand-computed unit tests for each report metric."""

import numpy as np
import pandas as pd
import pytest

from water_gte_eval.report_metrics import (
    auroc,
    auprc,
    bootstrap_over_events,
    discrimination,
    escalations_per_hour,
    event_detection,
    false_trust_at_budget,
    risk_coverage_points,
)


def test_risk_coverage_hand_computed():
    scores = [5, 4, 3, 2, 1]
    y = [1, 0, 0, 0, 0]
    pts = risk_coverage_points(scores, y, n_points=6)  # c = 0, .2, .4, .6, .8, 1.0
    # c=0: nothing escalated -> all 5 trusted, 1 anomalous -> err 0.2
    assert pts[0] == (0.0, 0.2)
    # c=0.2: k=1 -> top score (the anomalous one) escalated -> err 0.0
    assert pts[1] == (pytest.approx(0.2), 0.0)
    # c=1.0: all escalated -> trusted empty -> err 0.0
    assert pts[-1] == (1.0, 0.0)


def test_false_trust_at_budget_hand_computed():
    scores = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    y = [1, 1, 0, 0, 0, 0, 0, 0, 0, 1]
    r = false_trust_at_budget(scores, y, 0.2)
    assert r["n_escalated"] == 2 and r["n_trusted"] == 8
    assert r["n_wrong_trusted"] == 1  # only the score-0 window is anomalous among trusted
    assert r["false_trust_rate"] == pytest.approx(1 / 8)
    # budget 0.5: top 5 escalated (both pos at 9,8 escalated) -> trusted 5, wrong 1
    r = false_trust_at_budget(scores, y, 0.5)
    assert r["n_wrong_trusted"] == 1 and r["false_trust_rate"] == pytest.approx(1 / 5)


def test_auroc_auprc_hand_computed():
    assert auroc([1, 1, 0, 0], [0.9, 0.8, 0.3, 0.1]) == 1.0
    assert auroc([1, 0], [0.5, 0.5]) == 0.5  # tie
    assert auroc([1, 0], [0.1, 0.9]) == 0.0
    assert auroc([1, 1], [0.5, 0.4]) is None  # single class
    # AP: y=[1,0,1,0], s=[0.4,0.3,0.2,0.1] -> (1/1 + 2/3)/2 = 5/6
    assert auprc([1, 0, 1, 0], [0.4, 0.3, 0.2, 0.1]) == pytest.approx(5 / 6)
    assert auprc([1, 1, 0, 0], [0.9, 0.8, 0.3, 0.1]) == 1.0


def _ev_df():
    rows = []
    base = pd.Timestamp("2024-01-01T00:00:00")
    for i in range(3):  # evA: 3 hourly windows, flagged at the 2nd
        rows.append({"event_id": "evA", "window_id": f"a{i}", "start": base + pd.Timedelta(hours=i),
                     "flag": i == 1, "true_label": "attack"})
    for i in range(2):  # evB: never flagged
        rows.append({"event_id": "evB", "window_id": f"b{i}", "start": base + pd.Timedelta(hours=10 + i),
                     "flag": False, "true_label": "fault"})
    return pd.DataFrame(rows)


def test_event_detection_hand_computed():
    per_event, agg = event_detection(_ev_df(), min_detected_windows=1)
    assert agg["n_events"] == 2 and agg["n_detected"] == 1
    assert agg["fraction_detected"] == 0.5
    assert per_event["evA"]["detected"] and per_event["evA"]["delay_windows"] == 1
    assert per_event["evA"]["delay_hours"] == pytest.approx(1.0)
    assert per_event["evB"]["detected"] is False and per_event["evB"]["delay_hours"] is None
    assert agg["mean_delay_hours"] == pytest.approx(1.0)
    # min_detected_windows=2 -> evA (1 flag) not detected
    _, agg2 = event_detection(_ev_df(), min_detected_windows=2)
    assert agg2["fraction_detected"] == 0.0


def test_escalations_per_hour():
    flags = [True] * 6 + [False] * 4
    r = escalations_per_hour(flags, "2024-01-01T00:00:00", "2024-01-01T02:00:00")
    assert r == pytest.approx(3.0)
    assert escalations_per_hour(flags, "2024-01-01T00:00:00", "2024-01-01T00:00:00") is None


def test_discrimination_hand_computed():
    rows = []
    base = pd.Timestamp("2024-01-01T00:00:00")
    # fault event: windows flagged; process_change event: not flagged; scores separate them
    for i in range(3):
        rows.append({"event_id": "evF", "window_id": f"f{i}", "start": base + pd.Timedelta(hours=i),
                     "flag": True, "true_label": "fault", "score": 0.9 - 0.1 * i})
    for i in range(3):
        rows.append({"event_id": "evP", "window_id": f"p{i}", "start": base + pd.Timedelta(hours=10 + i),
                     "flag": False, "true_label": "process_change", "score": 0.1})
    df = pd.DataFrame(rows)
    d = discrimination(df, ("fault", "process_change"))
    assert d["window_auroc_fault_vs_process_change"] == 1.0
    assert d["event_detection_by_label"]["fault"]["fraction_detected"] == 1.0
    assert d["event_detection_by_label"]["process_change"]["fraction_detected"] == 0.0
    assert d["discrimination_gap"] == pytest.approx(1.0)


def test_bootstrap_over_events_resamples_events_not_windows():
    # events of sizes 1, 1, 10 -> fn = max event size in the resample must be 1 or 10
    # (window-level resampling would give means near 4)
    rows = []
    for e, size in (("e1", 1), ("e2", 1), ("e3", 10)):
        for i in range(size):
            rows.append({"event_id": e, "window_id": f"{e}{i}", "y": 1, "score": 1.0, "flag": True,
                         "start": "2024-01-01", "end": "2024-01-02", "true_label": "attack"})
    df = pd.DataFrame(rows)
    fn = lambda s: float(s.groupby("event_key" if "event_key" in s else "event_id").size().max()) if "event_key" in s else None
    # bootstrap_over_events adds event_key itself
    def fn_max(s):
        return float(s.groupby("event_key").size().max())
    b = bootstrap_over_events(df, fn_max, n_resamples=200, seed=1, ci=0.95)
    assert b["mean"] > 5  # event-level resampling: max is 10 whenever e3 is drawn
    # deterministic
    b2 = bootstrap_over_events(df, fn_max, n_resamples=200, seed=1, ci=0.95)
    assert b == b2
    assert b["ci_low"] <= b["mean"] <= b["ci_high"]
    # None handling
    b3 = bootstrap_over_events(df, lambda s: None, n_resamples=10, seed=1, ci=0.95)
    assert b3["mean"] is None and b3["n_dropped"] == 10
