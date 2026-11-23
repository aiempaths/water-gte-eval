"""MockEngine unit tests + registry checks."""

import dataclasses

import pytest

from gte_eval.engine_adapter import EngineError, make_engine
from gte_eval.engine_adapter.mock import MockEngine, fingerprint_window, sensor_fingerprint
from test_engine_contract import HISTORY, make_history, make_window


def test_fingerprint_stats():
    fp = sensor_fingerprint([1.0, 2.0, 3.0, 4.0])
    mean, std, slope, kurt, lag1 = fp
    assert mean == 2.5 and std == pytest.approx(1.118033988749895)
    assert slope == pytest.approx(1.0)
    assert kurt == pytest.approx(-1.36)  # excess kurtosis of a linear ramp
    assert -1.0 <= lag1 <= 1.0
    assert sensor_fingerprint([]) == [0.0] * 5
    assert sensor_fingerprint([5.0]) == [5.0, 0.0, 0.0, 0.0, 0.0]
    w = make_window(0, 1.0)
    assert fingerprint_window(w).shape == (10,)  # 2 sensors x 5 stats


def test_empty_history_is_unavailable():
    eng = MockEngine(k=3)
    r = eng.score(make_window(0, 1.0))
    assert r.verdict == "unavailable"
    assert r.closeness == float("inf") and r.agreement == 0.0 and r.neighbor_ids == []
    assert eng.verify_receipt(r) and eng.verify_chain()


def test_identical_history_trusted_far_routed():
    eng = MockEngine(k=3, closeness_max=2.0, agreement_min=0.6)
    eng.reset_history()
    eng.add_history([make_history(i, 1.0, "normal") for i in range(5)])
    r = eng.score(make_window(99, 1.0))
    assert r.verdict == "trust" and r.agreement == 1.0 and len(r.neighbor_ids) == 3
    assert r.closeness < 1e-9
    r_far = eng.score(make_window(98, 100.0))
    assert r_far.verdict == "route_to_human" and r_far.closeness > 2.0


def test_k_larger_than_history_and_mixed_labels():
    eng = MockEngine(k=10)
    eng.add_history([make_history(0, 1.0, "normal"), make_history(1, 2.0, "anomaly")])
    r = eng.score(make_window(2, 1.5))
    assert r.agreement == 0.5 and len(r.neighbor_ids) == 2


def test_determinism_across_instances():
    def run():
        eng = MockEngine(k=3)
        eng.add_history(HISTORY)
        return eng.score(make_window(7, 1.2))
    r1, r2 = run(), run()
    assert r1.receipt == r2.receipt
    assert dataclasses.replace(r1, latency_ms=0.0) == dataclasses.replace(r2, latency_ms=0.0)


def test_add_history_type_check():
    eng = MockEngine()
    with pytest.raises(EngineError, match="HistoryWindow"):
        eng.add_history([make_window(0, 1.0)])
    with pytest.raises(EngineError, match="score\\(\\) expects"):
        eng.score("not a window")


def test_registry():
    eng = make_engine({"type": "mock", "params": {"k": 3}})
    assert isinstance(eng, MockEngine) and eng.k == 3
    eng2 = make_engine({"type": "http", "params": {"base_url": "http://127.0.0.1:9"}})
    assert eng2.base_url == "http://127.0.0.1:9"
    with pytest.raises(EngineError, match="Unknown mock engine param"):
        make_engine({"type": "mock", "params": {"bogus": 1}})
    with pytest.raises(EngineError, match="Unknown engine type"):
        make_engine({"type": "real-gte"})
