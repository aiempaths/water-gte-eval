# Hostile self-audit (Prompt 11)

Adversarial review of the harness against the hard rules. Each finding has a
severity, file:line reference, and status. High-severity items were fixed in this
phase (commit after `61ec785`); the full suite and the example pipeline were
re-run afterwards (see bottom).

## Findings

### HIGH — fixed

1. **Cross-stage `next_window_state` leak (rolling mode).**
   `src/water_gte_eval/engine_run.py` `_run_mode` built `next_of` over the combined
   val+test list, so the last **val** window's "next window" was the first **test**
   window; in rolling mode that val window enters history while scoring test with
   the test window's label — a test label visible at scoring time.
   **Fix:** `next_of` is now computed per stage (`by_stage`), and `outcome_fn` gets
   the same-stage list; the last window of a stage falls back to its own label.
   **Regression test:** `tests/test_engine_run.py::test_next_window_state_never_crosses_stage_boundary`
   (fails on the pre-fix code, passes after; verified both ways).

### MEDIUM — fixed

2. **`eval` did not run the leakage checker on windows loaded from
   `windows_path`.** `prepare` runs `check_leakage`, but a hand-edited or
   third-party windows parquet could carry cross-stage time overlaps / shared
   event_ids into the runner silently.
   **Fix:** `run_engine_eval` now calls `check_leakage(train, val, test)` and raises
   `LeakageError` before scoring. **Test:** `test_eval_enforces_leakage_check_on_windows_file`.

### LOW — fixed

3. **`audit-chain` path resolution.** `audit.verify_run_receipts` re-loads the run's
   `config.yaml`; repo-relative `windows_path`/`prepare.source.path` only resolved
   when run from the repo root. **Fix:** fall back to the run dir's grandparent
   (`runs/<hash>/` → repo root). `src/water_gte_eval/audit.py`.

### Checked — clean (no change needed)

4. **Random splits:** none. `data/splits.py` is time/event-based only
   (`temporal_split`, `assign_stages` with event-snapped cuts + guard band);
   `check_leakage` raises on cross-stage overlap/shared event_id. The only RNGs
   are seeded `np.random.default_rng` used for bootstrap resampling, history
   subsampling (`cold_start`/`history.max_size`), poison flips, and robustness
   perturbations — never for splitting. Grep-verified: no `train_test_split`,
   no `shuffle` in split paths.
5. **History eligibility:** `_run_mode` adds a history window only when
   `h.end < w.start` (strict); rolling adds scored windows only after scoring;
   `next_window_state` knowability is guarded (`nxt.start >= w.start` → not added
   yet) and now per-stage (finding 1). Spy-engine tests cover all modes × outcomes
   (`test_no_test_label_ever_in_history_at_scoring`).
6. **Label-lag leakage:** `label_windows` applies the configured lag
   (`[start, end+lag)`) before `assign_stages`; event-snapped cuts + guard-band
   trimming + `check_leakage` keep lag-extended labels inside one stage, and sample
   data never crosses the `end < scored.start` boundary. Residual (accepted,
   user-configured) tradeoff: a history window's *label* may encode samples up to
   `end+lag` within its own stage. Documented in `docs/methodology.md`.
7. **Thresholds:** fit on val only, frozen, applied to test
   (`engine_run.run_engine_eval`, `baseline_eval.best_threshold`);
   `test_thresholds_chosen_on_val_never_test` proves the stored threshold equals the
   val-fitted one and test flags use it unchanged. Report PRIMARY comparison is
   threshold-free fixed budgets; the val section is marked "NOT evidence".
8. **EngineAdapter bypass:** all engine access goes through
   `gte_eval.engine_adapter` (`make_engine` registry) — `engine_run.py`,
   `runner.py` (legacy), `inject_eval.py`, `water_gte_eval/engine/__init__.py`
   re-export. No direct engine internals are touched outside the adapter package.
9. **Metrics on windows where events are required:** event-level metrics
   (`event_detection`, `build_event_records`) use only rows with an `event_id`;
   the bootstrap treats background windows as singleton units
   (`bootstrap_over_events`), never resampling within an event. No point-adjusted
   F1 anywhere; per-window and event-level metrics are separate.
10. **Report claims vs data:** the claims section uses the test split only, requires
    `n_events >= min_events_for_claims` for any method-comparison claim, and lists
    explicitly refused claims (val, single windows/scenarios, process_change as
    detections, generalization beyond the manifest). MockEngine runs are labeled
    "MOCK … NOT evidence" in the ablation section.
11. **Latency / wall-clock in `results.parquet`:** none. Per-window `latency_ms`
    lives only in `engine_log.parquet`; `results.parquet` rows contain config hash,
    git commit, mode/split/ids/labels, engine outputs + receipt, baseline
    scores/flags — no wall-clock fields, so re-runs are byte-identical (tested:
    `test_rerun_byte_identical_results_parquet`,
    `test_two_full_runs_byte_identical_results_and_receipts`). No `datetime.now()`
    anywhere in `src/` (grep-verified). `manifest.json`/`environment.json` contain
    hashes and version strings, not timestamps.
12. **Hard-coded dataset assumptions:** none in loaders. BATADAL ingest sniffs
    delimiter/header/timestamp/numeric/label columns (`ingest/detect.py`; the only
    heuristic is the documented `LABEL_NAME_RE = (attack|label)` name pattern, with
    `source.overrides` and printed `label_value_counts`); missing files stop with
    the exact path to create (`test_missing_dataset_file_stops_with_instructions`).
    SWaT/WADI template refuses placeholder column names.
13. **Tamper evidence:** receipts chain over (prev_receipt, window_id, inputs,
    outputs) with latency/raw excluded; `verify_chain()`/`find_chain_breaks()`
    (adapter level, first break = tampered row) and `audit-chain`
    (`verify_run_receipts`, run-artifact level, reports mode/position/window_id) —
    both tested, including a modified `results.parquet` row.

## Remaining risks (accepted, stated)

- The engine is a black box: the harness verifies *what it logs* (receipts, history
  eligibility via spy tests), not the engine's internal correctness.
- With `MockEngine`, every number is labeled MOCK and is not evidence about a real
  GTE; real-engine runs need the contract in `docs/engine_contract.md`.
- BATADAL has few events; the report refuses method comparisons below
  `min_events_for_claims` and shows (but does not interpret) CIs.
- WNTR/EPANET simulate distribution hydraulics only; `sim/toy_plant.py` is an
  uncalibrated toy; benchmark attacks are not real plant faults.
- `labeling.lag` can make a window's label depend on samples up to `lag` past the
  interval end (within its own stage; see finding 6).
- Report runtime grows with `bootstrap_n × methods` (~2–3 min at 1000 × 9 on the
  example); val section is point-only to keep it bounded.
- The Docker image could not be *built* in the development sandbox (no Docker
  daemon); equivalence was verified with a clean venv from `requirements-lock.txt`
  running the 5-command quickstart + full test collection. Build with
  `docker build -t water-gte-eval .` on a machine with Docker.

## Verification after fixes

- Full suite: re-run after the fixes (see commit message; N tests pass).
- Example pipeline re-run: `make-example-processed` → `prepare` → `eval` →
  `report` → `audit-chain` on the example run (all clean).
