"""EngineAdapter interface and the shared receipt-chain contract.

Types
-----
Window:         window_id, timestamps (ISO-8601 strings), sensors: {tag: [float, ...]}
HistoryWindow:  Window + outcome_label (str | int | None)
EngineResult:   window_id, closeness, agreement, verdict, neighbor_ids,
                receipt, latency_ms, raw

Receipt contract (tamper-evident hash chain)
--------------------------------------------
receipt_n = sha256(canonical_json({
    "prev_receipt": receipt_{n-1} (or "GENESIS"),
    "window_id":    window.window_id,
    "inputs":       {"window_id", "timestamps", "sensors"},
    "outputs":      {"closeness", "agreement", "verdict", "neighbor_ids"},
}))

canonical_json = json.dumps(sort_keys=True, separators=(",", ":")).
latency_ms and raw are deliberately NOT bound into the receipt, so the same
inputs always give a byte-identical receipt. Receipts are chain-position
dependent: scoring the same window twice on one adapter advances the chain.

The adapter base class keeps an append-only log of (window, result) pairs and
provides verify_receipt()/verify_chain() per this contract.
"""

from __future__ import annotations

import hashlib
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any

GENESIS = "GENESIS"
VERDICTS = ("trust", "route_to_human", "unavailable")


class EngineError(Exception):
    """Engine misconfiguration, contract violation, or API error."""


def canonical_json(obj: Any) -> str:
    """Deterministic JSON encoding used for receipts and hashing."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def _norm_timestamp(t: Any) -> str:
    return t.isoformat() if isinstance(t, datetime) else str(t)


@dataclass
class Window:
    """One multi-sensor window: id, sample timestamps, per-sensor values."""

    window_id: str
    timestamps: list | tuple
    sensors: dict

    def __post_init__(self) -> None:
        self.window_id = str(self.window_id)
        self.timestamps = tuple(_norm_timestamp(t) for t in self.timestamps)
        self.sensors = {
            str(tag): [float(v) for v in values]
            for tag, values in self.sensors.items()
        }

    def payload(self) -> dict:
        return {
            "window_id": self.window_id,
            "timestamps": list(self.timestamps),
            "sensors": {tag: list(vals) for tag, vals in self.sensors.items()},
        }


@dataclass
class HistoryWindow(Window):
    """A past labeled window stored in the engine's history."""

    outcome_label: str | int | None = None

    def payload(self) -> dict:
        p = super().payload()
        p["outcome_label"] = self.outcome_label
        return p


@dataclass(frozen=True)
class EngineResult:
    window_id: str
    closeness: float
    agreement: float
    verdict: str  # "trust" | "route_to_human" | "unavailable"
    neighbor_ids: list
    receipt: str
    latency_ms: float
    raw: dict = field(default_factory=dict)


def result_outputs(result: EngineResult) -> dict:
    """The output fields bound into the receipt (latency_ms and raw excluded)."""
    return {
        "closeness": result.closeness,
        "agreement": result.agreement,
        "verdict": result.verdict,
        "neighbor_ids": list(result.neighbor_ids),
    }


def compute_receipt(prev_receipt: str, window: Window, outputs: dict) -> str:
    payload = {
        "prev_receipt": prev_receipt,
        "window_id": window.window_id,
        "inputs": window.payload(),
        "outputs": outputs,
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


class EngineAdapter(ABC):
    """The ONLY interface to the GTE engine. See module docstring for the
    receipt contract."""

    engine_id: str = "abstract"

    def __init__(self) -> None:
        self._log: list[tuple[Window, EngineResult]] = []
        self._chain_head: str = GENESIS

    @abstractmethod
    def reset_history(self) -> None:
        """Clear history (and the local score log / receipt chain)."""

    @abstractmethod
    def add_history(self, windows: list[HistoryWindow]) -> None:
        """Store past labeled windows. The caller guarantees they all end
        strictly before the window about to be scored."""

    @abstractmethod
    def health(self) -> dict:
        """Engine status; must contain a "status" key."""

    @abstractmethod
    def _score_impl(self, window: Window) -> EngineResult:
        """Engine-specific scoring. latency_ms is set by score()."""

    def score(self, window: Window) -> EngineResult:
        """Score one window; records it in the log and measures latency."""
        if not isinstance(window, Window):
            raise EngineError(
                f"score() expects a gte_eval Window, got {type(window).__name__}"
            )
        t0 = time.perf_counter()
        result = self._score_impl(window)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        result = replace(result, latency_ms=float(latency_ms))
        if result.verdict not in VERDICTS:
            raise EngineError(f"invalid verdict {result.verdict!r}; expected one of {VERDICTS}")
        self._log.append((window, result))
        self._chain_head = result.receipt
        return result

    @property
    def log(self) -> list[tuple[Window, EngineResult]]:
        """Append-only log of scored (window, result) pairs."""
        return self._log

    @property
    def chain_head(self) -> str:
        return self._chain_head

    def _reset_log(self) -> None:
        self._log = []
        self._chain_head = GENESIS

    def _next_receipt(self, window: Window, outputs: dict) -> str:
        """Receipt for the next score, chained onto the current head."""
        return compute_receipt(self._chain_head, window, outputs)

    def verify_receipt(self, result: EngineResult) -> bool:
        """True iff `result` is the genuine chain entry for its window_id."""
        for i in range(len(self._log) - 1, -1, -1):
            window, logged = self._log[i]
            if logged.window_id == result.window_id:
                prev = self._log[i - 1][1].receipt if i > 0 else GENESIS
                return compute_receipt(prev, window, result_outputs(result)) == result.receipt
        return False

    def verify_chain(self) -> bool:
        """True iff every logged receipt re-derives correctly from GENESIS."""
        prev = GENESIS
        for window, result in self._log:
            if compute_receipt(prev, window, result_outputs(result)) != result.receipt:
                return False
            prev = result.receipt
        return True
