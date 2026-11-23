import numpy as np
import pandas as pd
import pytest

from water_gte_eval.engine import EngineError, MockEngine, make_engine
from water_gte_eval.engine.base import ScoreResult
from water_gte_eval.data.windows import Window


def make_window(i: int, base: float, start: str = "2024-01-01T00:00") -> Window:
    m = np.full((3, 2), base, dtype=np.float64)
    return Window(
        window_id=f"w{i:05d}",
        start=pd.Timestamp(start),
        end=pd.Timestamp(start) + pd.Timedelta("1h"),
        matrix=m,
        sensors=("a", "b"),
        label="normal",
        scenario="s",
    )


def test_empty_history_routes_to_human():
    eng = MockEngine(k=3)
    res = eng.score(make_window(0, 1.0))
    assert res.closeness is None and res.agreement is None
    assert res.verdict == "route_to_human"
    assert res.receipt and eng.verify_receipt(make_window(0, 1.0), res)


def test_identical_history_is_trusted():
    eng = MockEngine(k=3, closeness_max=2.0, agreement_min=0.6)
    for i in range(5):
        w = make_window(i, 1.0)
        eng.add_to_history(w, "normal")
    res = eng.score(make_window(99, 1.0))
    assert res.closeness is not None and res.closeness < 1e-9
    assert res.agreement == 1.0
    assert res.verdict == "trust"


def test_far_window_routes_to_human():
    eng = MockEngine(k=3, closeness_max=0.5, agreement_min=0.9)
    for i in range(5):
        eng.add_to_history(make_window(i, 1.0), "normal")
    res = eng.score(make_window(99, 100.0))
    assert res.verdict == "route_to_human"
    assert res.closeness > 0.5


def test_deterministic_across_instances():
    def run():
        eng = MockEngine(k=3)
        out = []
        for i in range(6):
            w = make_window(i, float(i))
            if i > 0:
                eng.add_to_history(make_window(i - 1, float(i - 1)), "normal" if i % 2 else "anomaly")
            out.append(eng.score(w))
        return out
    assert run() == run()


def test_receipt_detects_tampering():
    eng = MockEngine(k=2)
    eng.add_to_history(make_window(0, 1.0), "normal")
    w = make_window(1, 2.0)
    res = eng.score(w)
    assert eng.verify_receipt(w, res)
    tampered = ScoreResult(closeness=(res.closeness or 0) + 1.0, agreement=res.agreement,
                           verdict=res.verdict, receipt=res.receipt)
    assert not eng.verify_receipt(w, tampered)
    wrong_window = eng.verify_receipt(make_window(2, 2.0), res)
    assert not wrong_window


def test_k_larger_than_history_uses_all():
    eng = MockEngine(k=10)
    eng.add_to_history(make_window(0, 1.0), "normal")
    eng.add_to_history(make_window(1, 2.0), "anomaly")
    res = eng.score(make_window(2, 1.5))
    assert res.agreement == 0.5


def test_registry():
    eng = make_engine({"type": "mock", "params": {"k": 3}})
    assert isinstance(eng, MockEngine) and eng.k == 3
    with pytest.raises(EngineError, match="Unknown mock engine param"):
        make_engine({"type": "mock", "params": {"bogus": 1}})
    with pytest.raises(EngineError, match="Unknown engine type"):
        make_engine({"type": "real-gte"})
