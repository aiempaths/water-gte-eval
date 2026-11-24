# Documentation index

- [engine_contract.md](engine_contract.md) — the exact JSON/HTTP/receipt contract any
  GTE engine must satisfy, with sample request/response payloads, the `field_map`
  config, error behavior, and the reference fake server used in tests.
- [datasets.md](datasets.md) — where to place real datasets (BATADAL, SWaT, WADI),
  what the loaders expect, and exactly what happens when files are missing.
- [methodology.md](methodology.md) — leakage protections (time/event splits, history
  eligibility `end < scored.start`, within-stage `next_window_state`), threshold
  policy (val-only, frozen, then test), metric conventions, and known limitations.
- [adding_a_baseline.md](adding_a_baseline.md) — implement the `Baseline` interface,
  register it, add it to a run config.
- [adding_a_dataset.md](adding_a_dataset.md) — add a loader/ingest path for a new
  dataset without inventing schemas: detect, validate, document units.
- [../README.md](../README.md) — quickstart (5 commands to a report), pipeline
  overview, config reference, run/report outputs.
