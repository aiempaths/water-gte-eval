"""Contract tests that run against ANY EngineAdapter (mock or http+fake server).

Covers: determinism (same inputs -> byte-identical receipts), history isolation
(reset clears), receipt tamper checks (altering any logged field breaks chain
verification), and latency recording.
"""

from __future__ import annotations

import dataclasses
from string import hexdigits

import pandas as pd
import pytest

from fake_engine_server import create_app, start_server, stop_server
from gte_eval.engine_adapter import EngineResult, HistoryWindow, Window
from gte_eval.engine_adapter.http import HttpEngine
from gte_eval.engine_adapter.mock import MockEngine


def make_window(i: int, base: float, n: int = 6) -> Window:
    ts0 = pd.Timestamp("2024-01-01T00:00:00")
    ts = [(ts0 + pd.Timedelta(minutes=5 * j)).isoformat() for j in range(n)]
    return Window(
        window_id=f"w{i:03d}",
        timestamps=ts,
        sensors={"a": [base + 0.01 * j for j in range(n)], "b": [1.0] * n},
    )


def make_history(i: int, base: float, label) -> HistoryWindow:
    w = make_window(i, base)
    return HistoryWindow(
        window_id=w.window_id, timestamps=list(w.timestamps),
        sensors={k: list(v) for k, v in w.sensors.items()}, outcome_label=label,
    )


HISTORY = [make_history(0, 1.0, "normal"), make_history(1, 1.1, "normal"), make_history(2, 9.0, "anomaly")]


@pytest.fixture(scope="session")
def fake_server_url():
    url, server, thread = start_server(create_app())
    yield url
    stop_server(server, thread)


@pytest.fixture(params=["mock", "http"])
def adapter_factory(request, fake_server_url):
    if request.param == "mock":
        return lambda: MockEngine(k=3, closeness_max=2.0, agreement_min=0.6)
    return lambda: HttpEngine(base_url=fake_server_url, retries=2, timeout_s=5.0, backoff_s=0.01)


def _strip_latency(r: EngineResult) -> EngineResult:
    return dataclasses.replace(r, latency_ms=0.0)


def _score_with_history(factory) -> tuple:
    """Fresh adapter: reset -> add the shared history -> score. For http this
    also resets the shared fake server, so each run starts from GENESIS."""
    ad = factory()
    ad.reset_history()
    ad.add_history(HISTORY)
    return ad, ad.score(make_window(9, 1.05))


def test_same_inputs_give_byte_identical_receipts(adapter_factory):
    a, ra = _score_with_history(adapter_factory)
    b, rb = _score_with_history(adapter_factory)
    assert ra.receipt == rb.receipt
    assert _strip_latency(ra) == _strip_latency(rb)
    assert a.verify_chain() and b.verify_chain()


def test_score_twice_same_scores_chain_advances(adapter_factory):
    a = adapter_factory()
    a.reset_history()
    a.add_history(HISTORY)
    w = make_window(9, 1.05)
    r1, r2 = a.score(w), a.score(w)
    assert (r1.closeness, r1.agreement, r1.verdict, r1.neighbor_ids) == (
        r2.closeness, r2.agreement, r2.verdict, r2.neighbor_ids)
    assert r1.receipt != r2.receipt  # chain position differs; inputs identical


def test_reset_history_clears_isolation(adapter_factory):
    a = adapter_factory()
    a.reset_history()
    a.add_history([make_history(0, 1.0, "normal"), make_history(1, 1.1, "normal")])
    r_with = a.score(make_window(5, 1.05))
    assert r_with.verdict == "trust"
    a.reset_history()
    h = a.health()
    assert h["status"] == "ok"
    assert h.get("n_history_windows") == 0
    r_after = a.score(make_window(5, 1.05))
    assert r_after.verdict == "unavailable"
    fresh = adapter_factory()
    fresh.reset_history()
    r_fresh = fresh.score(make_window(5, 1.05))
    assert r_after.receipt == r_fresh.receipt
    assert _strip_latency(r_after) == _strip_latency(r_fresh)


def test_receipt_tamper_detection(adapter_factory):
    a = adapter_factory()
    a.reset_history()
    a.add_history(HISTORY)
    results = [a.score(make_window(i, 1.0 + 0.2 * i)) for i in range(4)]
    assert a.verify_chain()
    assert all(a.verify_receipt(r) for r in results)
    other_verdict = "route_to_human" if results[1].verdict != "route_to_human" else "trust"
    tamper_cases = [
        ("closeness", results[1].closeness + 1.0),
        ("agreement", results[1].agreement + 0.5 if results[1].agreement < 0.5 else 0.0),
        ("verdict", other_verdict),
        ("neighbor_ids", ["w999"]),
        ("window_id", "w999"),
        ("receipt", "0" * 64),
    ]
    for field, value in tamper_cases:
        tampered = dataclasses.replace(results[1], **{field: value})
        assert not a.verify_receipt(tampered), f"tampered {field} not detected"
    w0, r0 = a.log[0]
    a.log[0] = (w0, dataclasses.replace(r0, closeness=r0.closeness + 0.5))
    assert not a.verify_chain(), "altered log entry not detected by verify_chain"


def test_chain_break_identifies_tampered_row(adapter_factory):
    a = adapter_factory()
    a.reset_history()
    a.add_history(HISTORY)
    for i in range(4):
        a.score(make_window(i, 1.0 + 0.2 * i))
    assert a.verify_chain()
    assert a.find_chain_breaks() == []
    # tamper a logged OUTPUT field -> exactly that row is identified
    w2, r2 = a.log[2]
    a.log[2] = (w2, dataclasses.replace(r2, closeness=r2.closeness + 0.5))
    breaks = a.find_chain_breaks()
    assert breaks[0] == 2, f"first break must identify the tampered row, got {breaks}"
    assert not a.verify_chain()
    # tamper a stored RECEIPT field -> that row breaks first (later rows may cascade)
    b = adapter_factory()
    b.reset_history()
    b.add_history(HISTORY)
    for i in range(4):
        b.score(make_window(i, 1.0 + 0.2 * i))
    w1, r1 = b.log[1]
    b.log[1] = (w1, dataclasses.replace(r1, receipt="0" * 64))
    breaks2 = b.find_chain_breaks()
    assert breaks2[0] == 1, f"first break must identify the tampered row, got {breaks2}"


def test_result_shape_latency_and_health(adapter_factory):
    a = adapter_factory()
    a.reset_history()
    a.add_history(HISTORY)
    r = a.score(make_window(0, 1.0))
    assert r.verdict in ("trust", "route_to_human", "unavailable")
    assert isinstance(r.latency_ms, float) and r.latency_ms >= 0
    assert isinstance(r.neighbor_ids, list) and isinstance(r.raw, dict)
    assert len(r.receipt) == 64 and all(c in hexdigits for c in r.receipt)
    h = a.health()
    assert isinstance(h, dict) and "status" in h
