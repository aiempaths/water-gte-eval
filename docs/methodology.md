# Methodology: leakage protections, threshold policy, metric conventions, limitations

## Splits — never random

- `prepare` (`src/water_gte_eval/data/splits.py: assign_stages`) assigns
  time-ordered train/val/test stages. Split cuts **snap to event boundaries** so an
  entire attack/fault event stays in one split, and a guard band marks boundary
  windows of the earlier stage as `stage="dropped"` so no two stages overlap in
  time (required when `stride < window length`).
- `check_leakage` raises `LeakageError` listing every cross-stage time overlap or
  shared `event_id`; `prepare` runs it and prints the split summary (counts per
  split and per label). Tests include a deliberately leaky split that must fail.
- The only RNG uses anywhere are seeded and **not** used for splitting:
  bootstrap resampling over events, `cold_start`/`history.max_size` history
  subsampling, and robustness perturbations.

## History eligibility (engine)

- A window may enter the engine history only when its `end` is **strictly before**
  the scored window's `start` (`engine_run._run_mode`: `if h.end >= w.start:
  continue`). History is built from **train only**, plus (rolling mode) previously
  **scored** windows, added only **after** they are scored.
- `outcome: next_window_state` labels a history window with the NEXT window's
  label — and the "next window" is always taken **within the same stage**
  (`next_window_state` never crosses a stage boundary; crossing would let
  val/test labels into train history). The last window of a stage falls back to
  its own label. A rolling window whose next window has not happened yet
  (`next.start >= scored.start`) is not added yet — the label is not knowable.
- A spy-engine test (`tests/test_engine_run.py::
  test_no_test_label_ever_in_history_at_scoring`) runs all modes × outcomes and
  proves no test label is ever visible in history at scoring time.

## Labeling and label lag

- A window gets label L iff ≥ `min_overlap_frac` of its samples fall in
  `[start, end + lag)` of an L interval (`labeling.label_windows`); per-label
  overrides, priority for ties, `default_label` otherwise. The `lag` buffer exists
  because effects persist after the label ends. Labels are attached to windows;
  they never change which *samples* are in a window, so a lag cannot leak samples
  across the history-eligibility boundary (which is enforced on window times).
- Every labeled window carries its `event_id` so event-level metrics and the
  bootstrap group by event, never by window.

## Thresholds

- Baseline flag thresholds are chosen on **val only** (best F1), frozen, then
  applied to test (`engine_run` / `baseline_eval.best_threshold`). The report's
  PRIMARY comparison is threshold-free: fixed review budgets (5/10/20%) applied
  identically to engine and baselines by ranking. Scores are rank-normalized on
  val (empirical CDF) so engine and baseline scores share one [0,1] scale.
- The report's val section is marked "thresholds were fit here; NOT evidence";
  claims come from the test split only.

## Metrics

- Per-window and event-level only — **no point-adjusted F1**. Event detection:
  an event is detected when ≥ `min_detected_windows` of its windows are flagged;
  delay = first flagged window − event start.
- All CIs are bootstrap resamples **over events/scenarios** (background windows
  are singleton units), `bootstrap_n` from config — never over individual windows.
- `results.parquet` contains no wall-clock fields (per-window latency lives in
  the separate `engine_log.parquet`), so re-running a config gives byte-identical
  outputs (tested). Receipts are a sha256 chain over (prev_receipt, window_id,
  inputs, outputs); `audit-chain` re-derives them from a run dir and identifies
  tampered rows.

## Limitations (stated, not hidden)

- **WNTR/EPANET model distribution hydraulics only** — not in-plant coagulation,
  filtration, or water quality. The bundled chlorine trace is a simple
  first-order decay `C0·exp(−kt)`, not a WQ model. `sim/toy_plant.py` is a
  clearly-labeled, uncalibrated TOY model for in-plant processes.
- **Benchmark attacks differ from real plant faults.** BATADAL/SWaT/WADI attacks
  are network/sensor attacks on a benchmark; they are not evidence about faults
  in a real plant.
- **BATADAL has few events.** The report's claims section refuses method
  comparisons below `min_events_for_claims` independent events; CIs are shown
  but not interpreted there.
- The engine is a black box behind `EngineAdapter`; with `MockEngine` every
  number is labeled MOCK and is **not evidence** about a real GTE.
- `next_window_state` outcomes stay within a stage by design (see above), so the
  last window of a stage uses its own label.
