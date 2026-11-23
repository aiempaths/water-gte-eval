# water-gte-eval

Reproducible evaluation harness for a **Geometric Trust Engine (GTE)** — a black-box,
deterministic, training-free engine — on water-treatment/distribution telemetry.

The engine is accessed **only** through the `EngineAdapter` interface in
`src/gte_eval/engine_adapter/` (see [docs/engine_contract.md](docs/engine_contract.md)
for the exact JSON/HTTP contract). Two adapters ship: `MockEngine` (in-process,
deterministic) and `HttpEngine` (config-driven REST client). Contract tests in
`tests/test_engine_contract.py` run against ANY adapter; `tests/fake_engine_server.py`
is a FastAPI fake of the API for testing `HttpEngine`. Plug in the real engine by
implementing `EngineAdapter` and registering it in
`gte_eval/engine_adapter/registry.py`.

## Quickstart

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[test]"   # [test] adds fastapi+uvicorn for the fake server
.venv/bin/python -m water_gte_eval make-example-data        # synthetic example data (NOT real plant data)
.venv/bin/python -m water_gte_eval run configs/example_mock.yaml
.venv/bin/python -m pytest                                  # test suite
```

All commands run from the repo root; relative paths in configs resolve against it.

## Hard rules implemented here

- **One YAML config per experiment**; outputs go to `runs/<config_hash>/` where
  `config_hash` is the sha256 (first 16 hex chars) of the config's canonical JSON.
  Fixed seeds; outputs contain no wall-clock timestamps, so re-running a config
  reproduces byte-identical outputs (verified by tests).
- **Splits are by time and by scenario/event, never random.** `split.history_end`
  must be `<= split.eval_start`; `split.holdout_scenarios` removes scenarios from
  eval. At scoring time, history visible to the engine is restricted to windows
  ending **strictly before** the scored window's start (`data/splits.eligible_history`).
- **No invented data**: the config names file paths and column mappings; the loader
  validates them against the real files and errors with exactly what is missing.
  Supported file formats: `.parquet`, `.csv`.
- **No point-adjusted F1.** Reports are per-window (`metrics.json: window`,
  `predictions.parquet/csv`) and event-level (`metrics.json: event`, `events.csv`).
- **Bootstrap CIs over events or scenarios, never windows** (`bootstrap.unit`).

## Real data: what to place where

- `data/telemetry.parquet` (or `.csv`) — one row per sample: a timestamp column and
  one column per sensor.
- `data/events.parquet` (or `.csv`) — one row per labeled event/scenario: scenario id,
  start, end, label.

Then copy `configs/water_gte.yaml`, replace the `"<...>"` placeholders with your
real column names and split timestamps, and run
`python -m water_gte_eval run configs/water_gte.yaml`. If a file or column is
missing, the run stops and tells you the exact path and the columns it found.

## Config reference

See `configs/example_mock.yaml` for a fully commented example.

| key | meaning |
| --- | --- |
| `name`, `seed` | experiment name; fixed seed for the bootstrap RNG |
| `dataset.path/.format/.columns` | telemetry file + `timestamp`/`sensors` mapping |
| `events.path/.format/.columns` | events file + `scenario`/`start`/`end`/`label` mapping |
| `window.size/.step` | pandas durations, e.g. `1h` / `15min`; windows are `[start, start+size)` |
| `split.history_end/.eval_start/.eval_end/.holdout_scenarios` | time split + scenario hold-out |
| `labels.positive/.default` | positive label; label for windows outside any event |
| `engine.type/.params` | `mock` today; real engine once registered |
| `decision.mode` | `verdict` (flag when the engine verdict equals `route_verdict`) or `thresholds` (`max_closeness`/`min_agreement`; close+agreeing = trust, far or mixed = route to human) |
| `event_metrics.min_detected_windows` | flagged windows needed to count an event detected |
| `bootstrap.unit/.n_resamples/.ci/.seed` | `events`\|`scenarios`, count, level, seed |
| `output.runs_dir` | base output dir; the run goes to `<runs_dir>/<config_hash>/` |

## Window labeling

Each window gets the label/scenario of the event with the largest time overlap
(ties: earliest event start, then scenario id); windows overlapping no event get
`labels.default` and scenario `background`. Event-level metrics group eval windows
by their matched event; background windows count only in window-level metrics.

## Run outputs (`runs/<config_hash>/`)

- `config.yaml` — copy of the exact config used
- `manifest.json` — config hash, seed, engine id/params, data file sha256s, split,
  receipt-chain head, `receipts_verified`, output file hashes (no timestamps)
- `predictions.parquet` / `predictions.csv` — per-window: id, start, end, scenario,
  label, event, flag, closeness, agreement, verdict, receipt
- `events.csv` — per-event: windows, flagged windows, detected
- `metrics.json` — window-level (overall + per scenario), event-level, bootstrap CIs

## EngineAdapter interface (for the real engine)

```python
class EngineAdapter(ABC):
    def reset_history(self) -> None: ...
    def add_history(self, windows: list[HistoryWindow]) -> None: ...
    def score(self, window: Window) -> EngineResult: ...   # closeness, agreement, verdict, neighbor_ids, receipt, latency_ms, raw
    def health(self) -> dict: ...
```

`Window` = `{window_id, timestamps, sensors: {tag: [float,...]}}`;
`HistoryWindow` adds `outcome_label`. Verdicts: `trust` | `route_to_human` |
`unavailable`. Receipts are a sha256 chain over (prev_receipt, window_id, inputs,
outputs) — latency_ms/raw excluded so same inputs give byte-identical receipts.
The base class provides `verify_receipt`/`verify_chain`; the runner verifies the
whole chain and aborts on tampering. Full contract: [docs/engine_contract.md](docs/engine_contract.md).

`MockEngine` (`src/gte_eval/engine_adapter/mock.py`) fingerprints each sensor as
[mean, std, slope, excess kurtosis, lag-1 autocorrelation], z-scores by history
stats, kNN for closeness, majority-label fraction for agreement, thresholds for
the verdict. `HttpEngine` (`http.py`) maps endpoints and JSON field names entirely
from config (`configs/example_http.yaml`), with retries/timeouts and a clear
`EngineUnavailableError` when the API is unreachable.

## Layout

```
configs/    example_mock.yaml (runs today), example_http.yaml (needs your API), water_gte.yaml (real-data template)
data/       place real telemetry + events files here
docs/       engine_contract.md (exact JSON/HTTP/receipt contract)
examples/   synthetic example-data generator (NOT real plant data)
src/gte_eval/engine_adapter/   EngineAdapter, MockEngine, HttpEngine, registry (Prompt 2)
src/water_gte_eval/
  config.py            config loading, validation, config hash
  data/                loader (parquet/csv, schema checks), windows, splits
  engine/              re-exports the gte_eval.engine_adapter interface (single interface)
  metrics/             window-level, event-level, bootstrap CIs
  runner.py            one config -> runs/<config_hash>/
  cli.py               python -m water_gte_eval ...
tests/                 pytest suite: contract tests (any adapter), fake FastAPI server,
                       end-to-end determinism checks
```
