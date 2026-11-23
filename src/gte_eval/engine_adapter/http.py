"""HttpEngine: EngineAdapter over a local REST API.

Everything API-specific comes from config — never hard-coded:
  base_url            e.g. "http://127.0.0.1:8765" (PLACEHOLDER)
  endpoints           {reset, append, score, health} paths (PLACEHOLDER defaults)
  field_map           EngineResult field -> API response field name (PLACEHOLDER defaults)
  request_field_map   outgoing per-window JSON key names (PLACEHOLDER defaults)
  history_wrapper     top-level key for the append payload ("windows")
  score_wrapper       top-level key for the score payload ("window")
  timeout_s, retries, backoff_s

Unreachable API (connection error / timeout after all retries) raises
EngineUnavailableError with a clear message. HTTP >= 400 (except 503 on
/score) raises EngineError. A 503 from /score maps to verdict "unavailable".
Receipts are verified client-side per the contract in base.py.
"""

from __future__ import annotations

import time

import requests

from .base import (
    VERDICTS,
    EngineAdapter,
    EngineError,
    EngineResult,
    HistoryWindow,
    Window,
    result_outputs,
)


class EngineUnavailableError(EngineError):
    """The engine API could not be reached at all."""


DEFAULT_ENDPOINTS = {  # PLACEHOLDER defaults — override in config
    "reset": "/history/reset",
    "append": "/history/append",
    "score": "/score",
    "health": "/health",
}
DEFAULT_FIELD_MAP = {  # EngineResult field -> API response field; PLACEHOLDER defaults
    "window_id": "window_id",
    "closeness": "closeness",
    "agreement": "agreement",
    "verdict": "verdict",
    "neighbor_ids": "neighbor_ids",
    "receipt": "receipt",
}
DEFAULT_REQUEST_FIELD_MAP = {  # outgoing window JSON keys; PLACEHOLDER defaults
    "window_id": "window_id",
    "timestamps": "timestamps",
    "sensors": "sensors",
    "outcome_label": "outcome_label",
}


class HttpEngine(EngineAdapter):
    engine_id = "http"
    PARAMS = (
        "base_url",
        "endpoints",
        "field_map",
        "request_field_map",
        "history_wrapper",
        "score_wrapper",
        "timeout_s",
        "retries",
        "backoff_s",
    )

    def __init__(
        self,
        base_url: str,
        endpoints: dict | None = None,
        field_map: dict | None = None,
        request_field_map: dict | None = None,
        history_wrapper: str = "windows",
        score_wrapper: str = "window",
        timeout_s: float = 5.0,
        retries: int = 3,
        backoff_s: float = 0.2,
    ):
        super().__init__()
        if not base_url:
            raise EngineError("engine.params.base_url is required for type 'http'")
        self.base_url = str(base_url).rstrip("/")
        self.endpoints = {**DEFAULT_ENDPOINTS, **(endpoints or {})}
        self.field_map = {**DEFAULT_FIELD_MAP, **(field_map or {})}
        self.request_field_map = {**DEFAULT_REQUEST_FIELD_MAP, **(request_field_map or {})}
        self.history_wrapper = history_wrapper
        self.score_wrapper = score_wrapper
        self.timeout_s = float(timeout_s)
        self.retries = max(1, int(retries))
        self.backoff_s = float(backoff_s)
        self._session = requests.Session()
        self._n_appended = 0

    def _url(self, key: str) -> str:
        return self.base_url + self.endpoints[key]

    def _request(self, method: str, key: str, payload: dict | None = None):
        url = self._url(key)
        last_exc: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                resp = self._session.request(method, url, json=payload, timeout=self.timeout_s)
            except (requests.ConnectionError, requests.Timeout) as exc:
                last_exc = exc
                if attempt < self.retries:
                    time.sleep(self.backoff_s * attempt)
                continue
            if resp.status_code == 503 and key == "score":
                return resp
            if resp.status_code >= 400:
                raise EngineError(
                    f"engine API {method} {url} returned HTTP {resp.status_code}: {resp.text[:200]}"
                )
            return resp
        raise EngineUnavailableError(
            f"engine API unreachable at {url} after {self.retries} attempt(s) "
            f"(timeout={self.timeout_s}s): {last_exc}"
        )

    def _window_json(self, w: Window, with_label: bool) -> dict:
        rfm = self.request_field_map
        d = {
            rfm["window_id"]: w.window_id,
            rfm["timestamps"]: list(w.timestamps),
            rfm["sensors"]: {tag: list(vals) for tag, vals in w.sensors.items()},
        }
        if with_label:
            d[rfm["outcome_label"]] = w.outcome_label
        return d

    def reset_history(self) -> None:
        self._request("POST", "reset", payload={})
        self._reset_log()
        self._n_appended = 0

    def add_history(self, windows: list[HistoryWindow]) -> None:
        for w in windows:
            if not isinstance(w, HistoryWindow):
                raise EngineError(
                    f"add_history expects HistoryWindow items, got {type(w).__name__}"
                )
        payload = {self.history_wrapper: [self._window_json(w, with_label=True) for w in windows]}
        self._request("POST", "append", payload=payload)
        self._n_appended += len(windows)

    def health(self) -> dict:
        resp = self._request("GET", "health")
        try:
            data = resp.json() if resp.content else {}
        except ValueError as exc:
            raise EngineError(f"engine API GET {self._url('health')} returned non-JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise EngineError(f"engine API GET {self._url('health')} returned non-object JSON")
        out = {
            "status": data.get("status", "ok"),
            "engine": self.engine_id,
            "base_url": self.base_url,
            "n_appended_windows": self._n_appended,
            "n_scored": len(self._log),
            "chain_head": self._chain_head,
            "server_health": data,
        }
        if "n_history_windows" in data:
            out["n_history_windows"] = data["n_history_windows"]
        return out

    def _extract(self, data: dict, canonical: str):
        return data.get(self.field_map.get(canonical, canonical))

    def _score_impl(self, window: Window) -> EngineResult:
        payload = {self.score_wrapper: self._window_json(window, with_label=False)}
        resp = self._request("POST", "score", payload=payload)
        if resp.status_code == 503:
            outputs = {
                "closeness": float("inf"),
                "agreement": 0.0,
                "verdict": "unavailable",
                "neighbor_ids": [],
            }
            return EngineResult(
                window_id=window.window_id,
                closeness=outputs["closeness"],
                agreement=0.0,
                verdict="unavailable",
                neighbor_ids=[],
                receipt=self._next_receipt(window, outputs),
                latency_ms=0.0,
                raw={"http_status": 503},
            )
        try:
            data = resp.json()
        except ValueError as exc:
            raise EngineError(
                f"engine API POST {self._url('score')} returned non-JSON: {exc}"
            ) from exc
        if not isinstance(data, dict):
            raise EngineError(f"engine API POST {self._url('score')} returned non-object JSON")
        closeness = self._extract(data, "closeness")
        agreement = self._extract(data, "agreement")
        verdict = self._extract(data, "verdict")
        receipt = self._extract(data, "receipt")
        neighbor_ids = self._extract(data, "neighbor_ids") or []
        window_id = self._extract(data, "window_id") or window.window_id
        if verdict is None or receipt is None or agreement is None:
            missing = sorted(k for k, v in (("agreement", agreement), ("verdict", verdict), ("receipt", receipt)) if v is None)
            raise EngineError(
                f"engine API POST {self._url('score')} response is missing mapped field(s) "
                f"{missing}; check engine.params.field_map. Response keys: {sorted(data)}"
            )
        verdict = str(verdict)
        if verdict == "unavailable" and closeness is None:
            closeness = float("inf")  # strict JSON has no Infinity; null means unavailable
        if closeness is None:
            raise EngineError(
                f"engine API POST {self._url('score')} response is missing mapped field "
                f"'closeness' (null is only allowed when verdict='unavailable'); "
                f"check engine.params.field_map. Response keys: {sorted(data)}"
            )
        if verdict not in VERDICTS:
            raise EngineError(f"engine API returned invalid verdict {verdict!r}; expected one of {VERDICTS}")
            raise EngineError(
                f"engine API POST {self._url('score')} response is missing mapped field(s) "
                f"{sorted(k for k, v in (('closeness', closeness), ('agreement', agreement), ('verdict', verdict), ('receipt', receipt)) if v is None)}; "
                f"check engine.params.field_map. Response keys: {sorted(data)}"
            )
        return EngineResult(
            window_id=str(window_id),
            closeness=float(closeness),
            agreement=float(agreement),
            verdict=verdict,
            neighbor_ids=[str(x) for x in neighbor_ids],
            receipt=str(receipt),
            latency_ms=0.0,
            raw=data,
        )
