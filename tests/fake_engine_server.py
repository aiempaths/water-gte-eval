"""Tiny FastAPI fake of the GTE REST API, used to test HttpEngine without the
real engine. Implements the contract from docs/engine_contract.md by delegating
to a MockEngine internally. Loopback only (127.0.0.1, ephemeral port).

create_app options:
  engine_params      MockEngine params for the backing engine
  endpoints          override the endpoint paths (tests config-driven paths)
  response_aliases   rename /score response fields (tests field_map)
  score_503          make /score return HTTP 503 (tests verdict "unavailable")
"""

from __future__ import annotations

import socket
import threading
import time

from fastapi import FastAPI

from gte_eval.engine_adapter.base import EngineResult, HistoryWindow, Window
from gte_eval.engine_adapter.mock import MockEngine

DEFAULT_ENDPOINTS = {
    "reset": "/history/reset",
    "append": "/history/append",
    "score": "/score",
    "health": "/health",
}


def create_app(
    engine_params: dict | None = None,
    endpoints: dict | None = None,
    response_aliases: dict | None = None,
    score_503: bool = False,
) -> FastAPI:
    paths = {**DEFAULT_ENDPOINTS, **(endpoints or {})}
    aliases = response_aliases or {}
    engine = MockEngine(**(engine_params or {"k": 3, "closeness_max": 2.0, "agreement_min": 0.6}))
    app = FastAPI(title="fake-gte")

    def result_json(r: EngineResult) -> dict:
        d = {
            "window_id": r.window_id,
            # strict JSON has no Infinity: unavailable -> null (contract §5)
            "closeness": None if r.closeness == float("inf") else r.closeness,
            "agreement": r.agreement,
            "verdict": r.verdict,
            "neighbor_ids": list(r.neighbor_ids),
            "receipt": r.receipt,
        }
        return {aliases.get(k, k): v for k, v in d.items()}

    @app.post(paths["reset"])
    def reset() -> dict:
        engine.reset_history()
        return {"status": "ok"}

    @app.post(paths["append"])
    def append(body: dict) -> dict:
        wins = body.get("windows", [])
        hws = [
            HistoryWindow(
                window_id=w["window_id"],
                timestamps=w["timestamps"],
                sensors=w["sensors"],
                outcome_label=w.get("outcome_label"),
            )
            for w in wins
        ]
        engine.add_history(hws)
        return {"status": "ok", "accepted": len(hws)}

    @app.post(paths["score"])
    def score(body: dict):
        if score_503:
            from fastapi.responses import JSONResponse

            return JSONResponse(status_code=503, content={"status": "unavailable"})
        w = body["window"]
        r = engine.score(
            Window(window_id=w["window_id"], timestamps=w["timestamps"], sensors=w["sensors"])
        )
        return result_json(r)

    @app.get(paths["health"])
    def health() -> dict:
        return engine.health()

    return app


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def start_server(app) -> tuple[str, object, threading.Thread]:
    """Run a uvicorn server for `app` on 127.0.0.1:<ephemeral> in a daemon thread."""
    import uvicorn

    port = free_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 15
    while not server.started:
        if time.time() > deadline:
            raise RuntimeError("fake engine server did not start in time")
        time.sleep(0.05)
    return f"http://127.0.0.1:{port}", server, thread


def stop_server(server, thread) -> None:
    server.should_exit = True
    thread.join(timeout=10)
