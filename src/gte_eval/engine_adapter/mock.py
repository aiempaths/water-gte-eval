"""MockEngine: deterministic, training-free stand-in for the real GTE.

Fingerprint per sensor: [mean, std, slope, spikiness (excess kurtosis),
lag-1 autocorrelation], concatenated over sorted sensor tags. History
fingerprints are z-scored by running history mean/std; closeness = mean
euclidean distance to the k nearest history windows (ties by window_id);
agreement = fraction of those neighbors sharing the majority outcome label;
verdict = "trust" iff closeness <= closeness_max and agreement >=
agreement_min, else "route_to_human"; empty history -> "unavailable"
(closeness=inf, agreement=0.0, neighbor_ids=[]). Receipts follow the
sha256 chain contract in base.py, so same inputs give byte-identical receipts.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from .base import (
    EngineAdapter,
    EngineError,
    EngineResult,
    HistoryWindow,
    Window,
    result_outputs,
)


def sensor_fingerprint(values: list[float]) -> list[float]:
    x = np.asarray(values, dtype=np.float64)
    n = x.size
    if n == 0:
        return [0.0, 0.0, 0.0, 0.0, 0.0]
    mean = float(x.mean())
    std = float(x.std())
    if n >= 2:
        slope = float(np.polyfit(np.arange(n, dtype=np.float64), x, 1)[0])
        xc = x - mean
        m2 = float(np.mean(xc * xc))
        m4 = float(np.mean(xc**4))
        kurtosis = (m4 / (m2 * m2)) - 3.0 if m2 > 0 else 0.0
        lag1 = float(np.mean(xc[:-1] * xc[1:]) / m2) if m2 > 0 else 0.0
    else:
        slope = kurtosis = lag1 = 0.0
    return [mean, std, slope, kurtosis, lag1]


def fingerprint_window(window: Window) -> np.ndarray:
    parts: list[float] = []
    for tag in sorted(window.sensors):
        parts.extend(sensor_fingerprint(window.sensors[tag]))
    return np.asarray(parts, dtype=np.float64)


class MockEngine(EngineAdapter):
    engine_id = "mock"
    PARAMS = ("k", "closeness_max", "agreement_min")

    def __init__(self, k: int = 5, closeness_max: float = 2.0, agreement_min: float = 0.6):
        super().__init__()
        if int(k) < 1:
            raise EngineError("engine.params.k must be an integer >= 1")
        self.k = int(k)
        self.closeness_max = float(closeness_max)
        self.agreement_min = float(agreement_min)
        self._history: list[tuple[HistoryWindow, np.ndarray]] = []

    def reset_history(self) -> None:
        self._history = []
        self._reset_log()

    def add_history(self, windows: list[HistoryWindow]) -> None:
        for w in windows:
            if not isinstance(w, HistoryWindow):
                raise EngineError(
                    f"add_history expects HistoryWindow items, got {type(w).__name__}"
                )
            self._history.append((w, fingerprint_window(w)))

    def health(self) -> dict:
        return {
            "status": "ok",
            "engine": self.engine_id,
            "n_history_windows": len(self._history),
            "n_scored": len(self._log),
            "k": self.k,
            "closeness_max": self.closeness_max,
            "agreement_min": self.agreement_min,
            "chain_head": self._chain_head,
        }

    def _nearest(self, fp: np.ndarray, k: int) -> list[tuple[float, HistoryWindow]]:
        H = np.array([hfp for _, hfp in self._history], dtype=np.float64)
        mu = H.mean(axis=0)
        sd = H.std(axis=0)
        sd = np.where(sd < 1e-12, 1.0, sd)
        d = np.linalg.norm((H - mu) / sd - (fp - mu) / sd, axis=1)
        order = sorted(range(len(d)), key=lambda i: (float(d[i]), self._history[i][0].window_id))
        return [(float(d[i]), self._history[i][0]) for i in order[:k]]

    def _score_impl(self, window: Window) -> EngineResult:
        fp = fingerprint_window(window)
        if not self._history:
            outputs = {
                "closeness": float("inf"),
                "agreement": 0.0,
                "verdict": "unavailable",
                "neighbor_ids": [],
            }
            receipt = self._next_receipt(window, outputs)
            return EngineResult(
                window_id=window.window_id,
                closeness=outputs["closeness"],
                agreement=0.0,
                verdict="unavailable",
                neighbor_ids=[],
                receipt=receipt,
                latency_ms=0.0,
                raw={"reason": "empty_history", "n_history": 0},
            )
        nn = self._nearest(fp, self.k)
        neighbor_ids = [h.window_id for _, h in nn]
        distances = [d for d, _ in nn]
        closeness = float(np.mean(distances))
        labels = [h.outcome_label for _, h in nn]
        majority_label, majority_count = sorted(
            Counter(labels).items(), key=lambda kv: (-kv[1], str(kv[0]))
        )[0]
        agreement = float(majority_count) / len(nn)
        verdict = (
            "trust"
            if closeness <= self.closeness_max and agreement >= self.agreement_min
            else "route_to_human"
        )
        outputs = {
            "closeness": closeness,
            "agreement": agreement,
            "verdict": verdict,
            "neighbor_ids": neighbor_ids,
        }
        receipt = self._next_receipt(window, outputs)
        return EngineResult(
            window_id=window.window_id,
            closeness=closeness,
            agreement=agreement,
            verdict=verdict,
            neighbor_ids=neighbor_ids,
            receipt=receipt,
            latency_ms=0.0,
            raw={
                "n_history": len(self._history),
                "k": self.k,
                "neighbor_distances": distances,
                "neighbor_labels": [None if l is None else str(l) for l in labels],
                "majority_label": None if majority_label is None else str(majority_label),
            },
        )
