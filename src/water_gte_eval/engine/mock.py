"""MockEngine: deterministic, training-free stand-in for the real GTE.

Semantics (matching the black-box description):
- feature vector per window = per-sensor [mean, std], standardized by the
  running mean/std of the history features;
- closeness = mean euclidean distance to the k nearest history windows
  (lower = more familiar); ties broken by window id for determinism;
- agreement = fraction of those k nearest windows whose label equals the
  majority label among them (higher = more consistent);
- verdict = "trust" iff closeness <= closeness_max and agreement >=
  agreement_min, else "route_to_human"; with empty history the window is
  unfamiliar: closeness/agreement are None and the verdict routes to human;
- receipt = sha256 over a canonical JSON payload binding engine id, window,
  scores, verdict, and the previous receipt (hash chain, tamper-evident).
"""

from __future__ import annotations

import hashlib
from collections import Counter

import numpy as np

from ..config import canonical_json
from ..data.windows import Window
from .base import EngineAdapter, EngineError, ScoreResult

GENESIS = "GENESIS"


def _receipt(payload: dict) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _payload(engine_id: str, window: Window, result: ScoreResult, prev_receipt: str) -> dict:
    return {
        "engine_id": engine_id,
        "window_id": window.window_id,
        "window_start": window.start.isoformat(),
        "window_end": window.end.isoformat(),
        "closeness": result.closeness,
        "agreement": result.agreement,
        "verdict": result.verdict,
        "prev_receipt": prev_receipt,
    }


class MockEngine(EngineAdapter):
    engine_id = "mock"

    _PARAMS = ("k", "closeness_max", "agreement_min")

    def __init__(self, k: int = 5, closeness_max: float = 2.0, agreement_min: float = 0.6):
        if int(k) < 1:
            raise EngineError("engine.params.k must be >= 1")
        self.k = int(k)
        self.closeness_max = float(closeness_max)
        self.agreement_min = float(agreement_min)
        self.reset()

    def reset(self) -> None:
        # (window_id, start, end, feature_vector, label)
        self._history: list[tuple[str, "object", "object", np.ndarray, str]] = []
        # (window_id, result, prev_receipt) in scoring order
        self._log: list[tuple[str, ScoreResult, str]] = []

    def describe(self) -> dict:
        return {
            "type": self.engine_id,
            "params": {
                "k": self.k,
                "closeness_max": self.closeness_max,
                "agreement_min": self.agreement_min,
            },
        }

    @property
    def receipt_chain_head(self) -> str:
        return self._log[-1][1].receipt if self._log else GENESIS

    def add_to_history(self, window: Window, label: str) -> None:
        self._history.append(
            (window.window_id, window.start, window.end, window.feature_vector(), str(label))
        )

    def _nearest(self, fv: np.ndarray) -> list[tuple[float, str]]:
        if not self._history:
            return []
        H = np.array([h[3] for h in self._history], dtype=np.float64)
        mu = H.mean(axis=0)
        sd = H.std(axis=0)
        sd = np.where(sd < 1e-12, 1.0, sd)
        D = (H - mu) / sd
        d = np.linalg.norm(D - (fv - mu) / sd, axis=1)
        order = sorted(range(len(d)), key=lambda i: (float(d[i]), self._history[i][0]))
        return [(float(d[i]), self._history[i][4]) for i in order[: self.k]]

    def score(self, window: Window) -> ScoreResult:
        prev = self.receipt_chain_head
        nn = self._nearest(window.feature_vector())
        if not nn:
            result = ScoreResult(closeness=None, agreement=None, verdict="route_to_human",
                                 receipt="")
        else:
            closeness = float(np.mean([x[0] for x in nn]))
            labels = [x[1] for x in nn]
            majority_label, majority_count = sorted(
                Counter(labels).items(), key=lambda kv: (-kv[1], kv[0])
            )[0]
            agreement = float(majority_count) / len(nn)
            verdict = (
                "trust"
                if closeness <= self.closeness_max and agreement >= self.agreement_min
                else "route_to_human"
            )
            result = ScoreResult(closeness=closeness, agreement=agreement, verdict=verdict,
                                 receipt="")
        result = ScoreResult(
            closeness=result.closeness,
            agreement=result.agreement,
            verdict=result.verdict,
            receipt=_receipt(_payload(self.engine_id, window, result, prev)),
        )
        self._log.append((window.window_id, result, prev))
        return result

    def verify_receipt(self, window: Window, result: ScoreResult) -> bool:
        for wid, logged, prev in self._log:
            if wid == window.window_id:
                expected = _receipt(_payload(self.engine_id, window, result, prev))
                return expected == result.receipt and logged == result
        return False
