from water_gte_eval.metrics import EventRecord, bootstrap_event_metrics


def _records():
    recs = []
    for i, (key, scenario, label, n_flagged, n_windows) in enumerate([
        ("ev1|s1", "s1", "anomaly", 3, 4),
        ("ev2|s1", "s1", "normal", 0, 5),
        ("ev3|s2", "s2", "anomaly", 1, 2),
        ("ev4|s2", "s2", "normal", 2, 3),
        ("ev5|s3", "s3", "anomaly", 0, 3),
        ("ev6|s3", "s3", "normal", 1, 4),
    ]):
        recs.append(EventRecord(key, scenario, label, n_windows, n_flagged, n_flagged >= 1))
    return recs


def test_bootstrap_deterministic_for_fixed_seed():
    a = bootstrap_event_metrics(_records(), "events", 200, 0.95, seed=7, positive="anomaly")
    b = bootstrap_event_metrics(_records(), "events", 200, 0.95, seed=7, positive="anomaly")
    assert a == b
    assert a["unit"] == "events" and a["n_resamples"] == 200 and a["ci"] == 0.95
    for name in ("precision", "recall", "f1"):
        m = a["metrics"][name]
        assert m["ci_low"] <= m["ci_high"]
        assert m["n_valid"] + m["n_dropped"] == 200


def test_bootstrap_over_scenarios_resamples_whole_scenarios():
    a = bootstrap_event_metrics(_records(), "scenarios", 100, 0.9, seed=3, positive="anomaly")
    assert a["unit"] == "scenarios"
    assert a["metrics"]["recall"]["n_valid"] > 0


def test_bootstrap_rejects_window_unit():
    import pytest
    with pytest.raises(ValueError, match="events.*scenarios"):
        bootstrap_event_metrics(_records(), "windows", 10, 0.95, seed=1, positive="anomaly")
