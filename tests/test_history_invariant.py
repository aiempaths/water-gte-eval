"""The engine must never see history that overlaps the window being scored."""

from __future__ import annotations

from water_gte_eval.engine import MockEngine  # re-exported Prompt-2 MockEngine
from water_gte_eval.runner import run_experiment


class SpyEngine(MockEngine):
    def __init__(self):
        super().__init__(k=5)
        self.seen: list = []

    def _score_impl(self, window):
        self.seen.append(
            (window, [(h.window_id, h.timestamps[-1]) for h, _fp in self._history if h.timestamps])
        )
        return super()._score_impl(window)


def test_history_is_always_strictly_before_scored_window(config_path, monkeypatch):
    spy = SpyEngine()
    monkeypatch.setattr("water_gte_eval.runner.make_engine", lambda cfg: spy)
    run_experiment(config_path)
    assert spy.seen, "engine scored no windows"
    for window, history in spy.seen:
        assert window.timestamps, "scored window has no timestamps"
        for h_id, h_end_iso in history:
            # ISO-8601 naive timestamps compare lexicographically
            assert h_end_iso < window.timestamps[0], (
                f"history window {h_id} ends at {h_end_iso}, not strictly before "
                f"scored window start {window.timestamps[0]}"
            )
    sizes = [len(h) for _, h in spy.seen]
    assert sizes == sorted(sizes), "history must only grow over time"
