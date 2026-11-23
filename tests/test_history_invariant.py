"""The engine must never see history that overlaps the window being scored."""

from __future__ import annotations

import pandas as pd

from water_gte_eval.engine.mock import MockEngine
from water_gte_eval.runner import run_experiment


class SpyEngine(MockEngine):
    def __init__(self):
        super().__init__(k=5)
        self.seen: list = []

    def score(self, window):
        self.seen.append((window, [(h[1], h[2]) for h in self._history]))
        return super().score(window)


def test_history_is_always_strictly_before_scored_window(config_path, monkeypatch):
    spy = SpyEngine()
    monkeypatch.setattr("water_gte_eval.runner.make_engine", lambda cfg: spy)
    run_experiment(config_path)
    assert spy.seen, "engine scored no windows"
    for window, history in spy.seen:
        for h_start, h_end in history:
            assert h_end < window.start, (
                f"history window [{h_start}, {h_end}) overlaps scored window "
                f"[{window.start}, {window.end})"
            )
    sizes = [len(h) for _, h in spy.seen]
    assert sizes == sorted(sizes), "history must only grow over time"
