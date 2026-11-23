"""Injection tests: each fault modifies only its interval, records ground truth,
and injected labels match the injected intervals after windowing+labeling."""

import numpy as np
import pandas as pd
import pytest

from gte_eval.inject import FAULTS, inject_scenario, severity_sweep
from water_gte_eval.data import TelemetryColumns, build_sample_windows, label_windows


@pytest.fixture
def clean():
    idx = pd.date_range("2024-01-01", periods=48, freq="1h")
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "flow": 100 + rng.normal(0, 1, 48),
        "level": 5.0 + rng.normal(0, 0.05, 48),
        "pump_status": (np.arange(48) % 12 < 6).astype(float),
        "chlorine": 0.8 * np.exp(-0.02 * np.arange(48)) + rng.normal(0, 0.005, 48),
    }, index=idx)


S, E = "2024-01-02T06:00:00", "2024-01-02T12:00:00"  # hours 6..11


def _outside_unchanged(clean, mod, tag):
    mask = ~((mod.index >= S) & (mod.index < E))
    assert np.allclose(mod.loc[mask, tag], clean.loc[mask, tag])


def test_stuck_sensor(clean):
    mod, iv = inject_scenario(clean, [{"fault": "stuck_sensor", "start": S, "end": E,
                                       "severity": 1.0, "tags": ["flow"]}], seed=1)
    seg = mod.loc[(mod.index >= S) & (mod.index < E), "flow"]
    assert seg.nunique() == 1
    _outside_unchanged(clean, mod, "flow")
    _outside_unchanged(clean, mod, "level")
    assert iv.iloc[0]["label"] == "fault" and iv.iloc[0]["fault"] == "stuck_sensor"
    assert iv.iloc[0]["severity"] == 1.0 and iv.iloc[0]["event_id"] == "inj000"


def test_additive_drift_scales_with_severity(clean):
    out = severity_sweep(clean, "additive_drift", S, E, [0.5, 2.0], tags=["level"], seed=3)
    amps = []
    for sev, (mod, iv) in sorted(out.items()):
        seg = mod.loc[(mod.index >= S) & (mod.index < E), "level"]
        orig = clean.loc[(clean.index >= S) & (clean.index < E), "level"]
        amps.append((sev, float(np.max(np.abs(seg - orig)))))
    assert amps[0][1] < amps[1][1]  # monotonic in severity
    _outside_unchanged(clean, out[2.0][0], "level")


def test_constant_bias(clean):
    mod, _ = inject_scenario(clean, [{"fault": "constant_bias", "start": S, "end": E,
                                      "severity": 1.0, "tags": ["flow"]}], seed=1)
    seg = mod.loc[(mod.index >= S) & (mod.index < E), "flow"]
    orig = clean.loc[(clean.index >= S) & (clean.index < E), "flow"]
    diff = seg - orig
    assert diff.std() < 1e-9  # constant
    assert abs(diff.iloc[0] - clean["flow"].std()) < 1e-9  # severity * scale


def test_spikes(clean):
    mod, _ = inject_scenario(clean, [{"fault": "spikes", "start": S, "end": E,
                                      "severity": 0.5, "tags": ["chlorine"]}], seed=7)
    seg = mod.loc[(mod.index >= S) & (mod.index < E), "chlorine"]
    orig = clean.loc[(clean.index >= S) & (clean.index < E), "chlorine"]
    n_big = int((np.abs(seg - orig) > 0.03).sum())
    n = len(seg)
    assert n_big == min(max(1, int(round(0.5 * 20))), n)  # severity-scaled spike count


def test_rising_noise(clean):
    mod, _ = inject_scenario(clean, [{"fault": "rising_noise", "start": S, "end": E,
                                      "severity": 2.0, "tags": ["flow"]}], seed=5)
    seg = mod.loc[(mod.index >= S) & (mod.index < E), "flow"]
    orig = clean.loc[(clean.index >= S) & (clean.index < E), "flow"]
    resid = np.abs(seg - orig)
    assert resid.iloc[-3:].mean() > resid.iloc[:3].mean()  # noise grows over interval


def test_replay_uses_earlier_data(clean):
    mod, _ = inject_scenario(clean, [{"fault": "replay", "start": S, "end": E,
                                      "severity": 1.0, "tags": ["flow"],
                                      "params": {"replay_offset_samples": 24}}], seed=1)
    a = clean.index.get_loc(pd.Timestamp(S))
    n = (pd.Timestamp(E) - pd.Timestamp(S)) // pd.Timedelta("1h")
    seg = mod.loc[(mod.index >= S) & (mod.index < E), "flow"].to_numpy()
    src = clean["flow"].iloc[a - 24: a - 24 + n].to_numpy()
    assert np.allclose(seg, src)
    _outside_unchanged(clean, mod, "flow")


def test_delayed_dropped(clean):
    mod, _ = inject_scenario(clean, [{"fault": "delayed_dropped", "start": S, "end": E,
                                      "severity": 1.0, "tags": ["flow"]}], seed=2)
    seg = mod.loc[(mod.index >= S) & (mod.index < E), "flow"]
    assert seg.isna().sum() > 0  # dropped samples
    # delay: value at a+shift equals original at a (where not dropped)
    orig = clean["flow"]
    a = clean.index.get_loc(pd.Timestamp(S))
    shift = 4
    assert mod["flow"].iloc[a + shift] == orig.iloc[a] or pd.isna(mod["flow"].iloc[a + shift])


def test_actuator_fault_inconsistent(clean):
    mod, iv = inject_scenario(clean, [{"fault": "actuator_fault", "start": S, "end": E,
                                       "severity": 1.0,
                                       "params": {"pump_tag": "pump_status", "flow_tag": "flow",
                                                  "mode": "pump_off_flow_high"}}], seed=1)
    seg = mod.loc[(mod.index >= S) & (mod.index < E)]
    assert (seg["pump_status"] == 0).all()          # pump off...
    assert np.allclose(seg["flow"], clean.loc[seg.index, "flow"])  # ...but flow unchanged: inconsistent
    assert iv.iloc[0]["label"] == "fault"


def test_legitimate_disturbances_are_process_change_not_fault(clean):
    mod, iv = inject_scenario(clean, [
        {"fault": "demand_surge", "start": S, "end": E, "severity": 0.8, "tags": ["flow"]},
        {"fault": "tank_refill", "start": S, "end": E, "severity": 0.6, "tags": ["level"]},
    ], seed=1)
    assert set(iv["label"]) == {"process_change"}
    assert "fault" not in set(iv["label"])
    seg = mod.loc[(mod.index >= S) & (mod.index < E), "flow"]
    orig = clean.loc[(clean.index >= S) & (clean.index < E), "flow"]
    assert np.allclose(seg, orig * 1.8)  # surge scaled by 1 + severity


def test_all_fault_types_registered():
    assert sorted(FAULTS) == sorted([
        "stuck_sensor", "additive_drift", "constant_bias", "spikes", "rising_noise",
        "replay", "delayed_dropped", "actuator_fault", "demand_surge", "tank_refill"])


def test_injected_labels_match_injected_intervals(clean):
    injections = [
        {"fault": "constant_bias", "start": S, "end": E, "severity": 1.0, "tags": ["flow"],
         "label": "fault", "event_id": "evA"},
        {"fault": "demand_surge", "start": "2024-01-02T14:00:00", "end": "2024-01-02T22:00:00",
         "severity": 0.7, "tags": ["flow"], "label": "process_change", "event_id": "evB"},
    ]
    mod, intervals = inject_scenario(clean, injections, seed=11)
    windows = build_sample_windows(mod, TelemetryColumns(timestamp="timestamp",
                                                         sensors=tuple(mod.columns)),
                                   length=6, stride=6)
    label_windows(windows, intervals, min_overlap_frac=0.5, lag=pd.Timedelta("0h"))
    # every window carrying an event_id has that interval's label
    by_event = {}
    for w in windows:
        if w.event_id is not None:
            by_event.setdefault(w.event_id, set()).add(w.label)
    for _, row in intervals.iterrows():
        assert row["event_id"] in by_event, f"no window got event_id {row['event_id']}"
        assert by_event[row["event_id"]] == {row["label"]}, \
            f"label mismatch for {row['event_id']}: {by_event[row['event_id']]} vs {row['label']}"
    # intervals table itself records ground truth
    assert set(intervals["event_id"]) == {"evA", "evB"}
    assert list(intervals["label"]) == ["fault", "process_change"]


def test_determinism(clean):
    inj = [{"fault": "spikes", "start": S, "end": E, "severity": 0.5, "tags": ["flow"]},
           {"fault": "rising_noise", "start": S, "end": E, "severity": 1.0, "tags": ["level"]}]
    m1, i1 = inject_scenario(clean, inj, seed=99)
    m2, i2 = inject_scenario(clean, inj, seed=99)
    pd.testing.assert_frame_equal(m1, m2)
    pd.testing.assert_frame_equal(i1, i2)


def test_unknown_fault_rejected(clean):
    with pytest.raises(ValueError, match="unknown fault"):
        inject_scenario(clean, [{"fault": "magic", "start": S, "end": E}], seed=1)
