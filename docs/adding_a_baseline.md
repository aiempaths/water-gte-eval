# How to add a baseline

Baselines live in `src/gte_eval/baselines/` and share one interface
(`baselines/base.py`):

```python
class Baseline(ABC):
    name: str
    PARAMS: tuple = ()                      # tunable params (validated from config)
    def fit(self, train_windows: list[Window], labels: list) -> None: ...
    def score(self, window: Window) -> float: ...        # untrust, higher = worse
    def score_windows(self, windows) -> list[float]: ...  # provided by the base
```

`Window` here is `gte_eval.engine_adapter.Window` (window_id, timestamps,
sensors: {tag: [float, ...]}). Shared feature helpers are in
`baselines/features.py` (`summary_stats` = per-tag mean/std/min/max/slope;
`raw_downsampled`).

Steps:

1. Create `src/gte_eval/baselines/my_baseline.py` implementing `Baseline`.
   Seed everything (`np.random.default_rng(seed)`) — baselines must be
   deterministic for a fixed config.
2. Register it in `src/gte_eval/baselines/registry.py` (`REGISTRY` /
   `make_baseline`), and export it from `baselines/__init__.py`.
3. Add it to a run config: `baselines: [..., my_baseline]` plus optional
   `baseline_params: {my_baseline: {...}}` in `configs/run/*.yaml`.
4. Add unit tests on synthetic data where the answer is known (see
   `tests/test_baselines.py`), including determinism (fit twice → identical
   scores).

The baseline then automatically appears in `eval` results, the report (ranked
side by side with the engine, with event-bootstrap CIs), and can join injection
severity sweeps via `sweep.detectors` in an inject config.
