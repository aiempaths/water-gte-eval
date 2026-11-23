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

## Ingest (raw -> processed)

```bash
python -m water_gte_eval make-example-raw                       # synthetic raw csvs (NOT real data)
python -m water_gte_eval ingest configs/ingest/example.yaml     # prints a data-quality report
python -m water_gte_eval ingest configs/ingest/batadal.yaml     # real BATADAL once files are placed
```

- **BATADAL** (`source.type: batadal_dir`, path `data/raw/batadal/`): every
  `.csv/.dat/.txt/.tsv` file is schema-**detected** — delimiter, header,
  timestamp column (ISO strings or epoch s/ms/us/ns), numeric tag columns, and
  attack-label columns (names matching /attack|label/i). No file or column names
  are assumed; `source.overrides` can override detection. Missing dir -> the run
  stops and says exactly what to place where. What I found so far: **no files yet**
  (`data/raw/batadal/` is empty) — place the raw BATADAL (C-Town) files there.
- **Generic CSV** (`source.type: csv`, for SWaT/WADI once access is granted):
  configurable `time_column`, `tag_groups`, `label_column`, delimiter (sniffed),
  `resample`+`resample_agg` to unify sampling rates, and `missing`
  (interpolate/ffill/bfill/drop) with optional `missing_limit`.
- **Validator**: units documented per tag in `configs/tags.yaml` (unit-less tags
  reported, errors when `require_units: true`, optional min/max range checks),
  monotonic time, duplicate timestamps, gap report vs expected step (inferred if
  unset), flatline report per tag, missing-value counts. The report is printed
  and stored in the manifest.
- Output: tidy parquet (DatetimeIndex, one float column per tag, label columns,
  `source_file`) into `output.dir` (default `data/processed/`) plus a
  `manifest.json` with source file sha256s, row counts, date ranges, detected
  schemas, and quality reports. Deterministic: no wall-clock fields.

## Prepare: windows, labels, splits

```bash
python -m water_gte_eval make-example-processed
python -m water_gte_eval prepare configs/prepare/example.yaml   # prints counts per split and per label
python -m water_gte_eval prepare configs/prepare/batadal.yaml   # once data/processed/ exists
```

- **Windowing** (`data/windowing.py`): sliding windows by length/stride **in
  samples** over the tidy frame; each `Window` has window_id, half-open
  [start, end), per-tag arrays (`sensor_arrays()`), a `tag_group`, and later a
  `stage`. The same Window objects feed the engine and all baselines.
- **Labeling** (`data/labeling.py`): labeled intervals (explicit file, or derived
  from runs of the sample-level label column) with columns event_id/start/end/label.
  A window gets label L when >= `min_overlap_frac` of its samples fall in
  [start, end + `lag`) — per-label overrides supported; `priority` breaks ties;
  unlabeled windows get `default_label`. Every labeled window carries its
  `event_id` (for event-level bootstrapping).
- **Splits** (`data/splits.py`): `assign_stages` gives time-ordered
  train/val/test; cuts snap so an entire event stays in one split, and a guard
  band drops boundary windows of the earlier stage so no two stages overlap in
  time (needed when stride < length). `check_leakage` raises `LeakageError`
  listing every cross-stage time overlap or shared event_id. `prepare` runs the
  checker and prints window counts per split and per label.

## Baselines

`src/gte_eval/baselines/` implements a common interface —
`fit(train_windows, labels)` / `score(window) -> float` untrust score
(higher = less trustworthy) — so all baselines share the same metrics code:

| name | idea |
| --- | --- |
| `zscore` | max \|z\| over per-tag summary stats vs train mean/std (limit checks) |
| `iforest` | IsolationForest on summary stats; score = −decision_function |
| `autoencoder` | reconstruction MSE of a small sklearn MLPRegressor bottleneck on summary stats |
| `knn_raw` | mean distance to k nearest train windows on RAW downsampled values |
| `conformal` | split-conformal: logistic regression on summary stats, contiguous fit/cal split; score = P(positive); `conformal_threshold(alpha)` exposed |
| `simple_knn_trust` | the engine's two-score design (closeness + agreement, `score_parts()`) on PLAIN summary stats (mean, std, min, max, slope) — tests whether the geometric fingerprint adds anything |

Driver: `python -m water_gte_eval baselines configs/baselines/example.yaml`
(after `prepare`). Fits on the train stage, picks the flag threshold on val
(best F1, or conformal quantile), scores test, and writes `runs/<config_hash>/`
with `baselines.json` (per baseline: threshold, val metrics, test ROC-AUC,
window-level + event-level metrics, bootstrap CIs over events) and
`baseline_scores(_wide).parquet`. All seeded and deterministic; a perf smoke
runs all six on 2400 windows in seconds.

## Simulation, injection, severity sweeps

```bash
python -m water_gte_eval sim-wntr configs/sim/wntr_example.yaml   # WNTR on bundled EPANET Net3
python -m water_gte_eval sim-toy configs/sim/toy_example.yaml     # clearly-labeled TOY plant model
python -m water_gte_eval inject-demo configs/inject/example.yaml  # plots + severity curve -> reports/
```

- **`src/gte_eval/sim/wntr_sim.py`**: hydraulic telemetry from WNTR running the
  **bundled EPANET Net3** (`<wntr pkg>/library/networks/Net3.inp`, path recorded
  in the manifest with its sha256), one random 24-step demand pattern per seed,
  reporting tank levels, junction pressures, pump flows/status, pipe flows,
  total demand, plus a **simple first-order chlorine decay trace**
  (`C0*exp(-k*t)`) — not a water-quality model. WNTR/EPANET model distribution
  hydraulics only; they do **not** model in-plant coagulation or filtration.
- **`src/gte_eval/sim/toy_plant.py`**: clearly labeled TOY in-plant model (tank
  balance, chlorine-contact CSTR, filter headloss growth with noise + backwash,
  pump hysteresis) for the processes WNTR doesn't cover. Not calibrated.
- **`src/gte_eval/inject/`**: fault injection into clean telemetry with a
  severity parameter and recorded ground truth (event_id/start/end/label/
  fault/severity/tags): stuck_sensor, additive_drift, constant_bias, spikes,
  rising_noise, replay, delayed_dropped, actuator_fault (pump vs flow
  inconsistency), and legitimate disturbances demand_surge / tank_refill
  labeled **process_change, not fault**. `severity_sweep` applies one fault at
  many severities. Feeding the intervals to `label_windows` makes injected
  labels match the injected intervals exactly (tested).
- **`inject-demo`**: plots a few injected examples (clean vs injected, interval
  shaded) to `reports/injected_*.png`, and runs a severity sweep through a
  detector baseline fit on clean windows -> `reports/severity_curve.png/.json`
  (event recall vs severity) + `injected_intervals.parquet` + `inject_demo.json`.

## Layout

```
configs/    experiment + ingest + prepare + baselines + sim + inject configs, tags.yaml
data/       raw/ (place BATADAL/SWaT files here), processed/ (generated parquet + manifest.json)
docs/       engine_contract.md (exact JSON/HTTP/receipt contract)
examples/   synthetic example-data generators (NOT real plant data)
src/gte_eval/engine_adapter/   EngineAdapter, MockEngine, HttpEngine, registry (Prompt 2)
src/gte_eval/baselines/        common Baseline interface + 6 baselines + registry
src/gte_eval/inject/           fault injection + scenarios + severity sweeps
src/gte_eval/sim/              wntr_sim (bundled Net3) + toy_plant (labeled toy)
src/water_gte_eval/
  config.py            experiment config loading, validation, config hash
  data/                loader (parquet/csv, schema checks), windows, splits
  engine/              re-exports the gte_eval.engine_adapter interface (single interface)
  ingest/              detect, batadal, generic_csv, validator, pipeline, ingest config
  prepare.py           processed parquet -> windows + labels + staged split + leakage check
  baseline_eval.py     windows parquet -> baselines -> runs/<config_hash>/ shared metrics
  inject_eval.py       inject-demo: example plots + severity curve -> reports/
  data/                loader, windows/windowing, labeling, splits + leakage checker
  metrics/             window-level, event-level, bootstrap CIs
  runner.py            one config -> runs/<config_hash>/
  cli.py               python -m water_gte_eval {run, ingest, prepare, baselines,
                       sim-wntr, sim-toy, inject-demo, make-example-*}
reports/               inject-demo plots/curves (generated)
tests/                 pytest suite: engine contract tests (any adapter), fake FastAPI
                       server, ingest tests (gaps/duplicates/unit-less/flatline), e2e determinism
```
