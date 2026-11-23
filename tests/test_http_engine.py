"""HttpEngine tests against the FastAPI fake server (loopback) + failure modes."""

import dataclasses

import pytest

from fake_engine_server import create_app, free_port, start_server, stop_server
from gte_eval.engine_adapter import EngineUnavailableError, EngineError, HistoryWindow
from gte_eval.engine_adapter.http import HttpEngine
from test_engine_contract import HISTORY, make_history, make_window


@pytest.fixture
def server_url():
    url, server, thread = start_server(create_app())
    yield url
    stop_server(server, thread)


def _engine(url, **kw):
    params = {"base_url": url, "retries": 2, "timeout_s": 5.0, "backoff_s": 0.01}
    params.update(kw)
    return HttpEngine(**params)


def test_roundtrip_against_fake_server(server_url):
    eng = _engine(server_url)
    eng.reset_history()
    eng.add_history(HISTORY)
    r = eng.score(make_window(9, 1.05))
    assert r.verdict == "trust"
    assert r.window_id == "w009"
    assert len(r.neighbor_ids) == 3
    assert eng.verify_chain() and eng.verify_receipt(r)
    h = eng.health()
    assert h["status"] == "ok" and h["n_history_windows"] == 3


def test_unreachable_api_raises_clear_error():
    port = free_port()  # bound then closed -> nothing listening
    eng = HttpEngine(base_url=f"http://127.0.0.1:{port}", retries=2, timeout_s=0.5, backoff_s=0.01)
    with pytest.raises(EngineUnavailableError) as exc:
        eng.score(make_window(0, 1.0))
    msg = str(exc.value)
    assert "unreachable" in msg and f"127.0.0.1:{port}" in msg and "2 attempt" in msg


def test_custom_endpoints_from_config():
    endpoints = {"reset": "/r", "append": "/a", "score": "/s", "health": "/h"}
    url, server, thread = start_server(create_app(endpoints=endpoints))
    try:
        eng = _engine(url, endpoints=endpoints)
        eng.reset_history()
        eng.add_history(HISTORY)
        r = eng.score(make_window(9, 1.05))
        assert r.verdict == "trust" and eng.health()["status"] == "ok"
    finally:
        stop_server(server, thread)


def test_field_map_renames_response_fields():
    aliases = {
        "window_id": "id", "closeness": "dist", "agreement": "agree",
        "verdict": "decision", "neighbor_ids": "neighbors", "receipt": "receipt_hash",
    }
    url, server, thread = start_server(create_app(response_aliases=aliases))
    try:
        field_map = {
            "window_id": "id", "closeness": "dist", "agreement": "agree",
            "verdict": "decision", "neighbor_ids": "neighbors", "receipt": "receipt_hash",
        }
        eng = _engine(url, field_map=field_map)
        eng.reset_history()
        eng.add_history(HISTORY)
        r = eng.score(make_window(9, 1.05))
        assert r.verdict == "trust" and r.window_id == "w009" and eng.verify_receipt(r)
    finally:
        stop_server(server, thread)


def test_request_field_map_renames_outgoing_fields():
    # server reads fixed canonical names; rename outgoing keys and confirm the
    # server sees missing keys -> the fake server would KeyError -> HTTP 500.
    url, server, thread = start_server(create_app())
    try:
        eng = _engine(url, request_field_map={"window_id": "wid", "timestamps": "timestamps",
                                              "sensors": "sensors", "outcome_label": "outcome_label"})
        eng.reset_history()
        with pytest.raises(EngineError, match="HTTP 500"):
            eng.add_history(HISTORY)
    finally:
        stop_server(server, thread)


def test_score_503_maps_to_unavailable():
    url, server, thread = start_server(create_app(score_503=True))
    try:
        eng = _engine(url)
        eng.reset_history()
        r = eng.score(make_window(0, 1.0))
        assert r.verdict == "unavailable" and r.closeness == float("inf")
        assert eng.verify_chain() and eng.verify_receipt(r)
    finally:
        stop_server(server, thread)


def test_http_determinism_matches_mock(server_url):
    from gte_eval.engine_adapter.mock import MockEngine

    mock = MockEngine(k=3, closeness_max=2.0, agreement_min=0.6)
    mock.reset_history()
    mock.add_history(HISTORY)
    r_mock = mock.score(make_window(9, 1.05))

    eng = _engine(server_url)
    eng.reset_history()
    eng.add_history(HISTORY)
    r_http = eng.score(make_window(9, 1.05))

    assert r_mock.receipt == r_http.receipt  # same contract chain on both sides
    assert dataclasses.replace(r_mock, latency_ms=0.0, raw=r_http.raw) == dataclasses.replace(
        r_http, latency_ms=0.0, raw=r_http.raw)
