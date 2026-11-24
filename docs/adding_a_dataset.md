# How to add a dataset

Rules: **never invent URLs, column names, or formats.** Inspect real files; if a
file is missing, stop and say exactly what to place where.

## 1. Decide the ingest path

- **BATADAL-style directory of mixed files** → the schema-detecting directory
  loader (`ingest/batadal.py` + `ingest/detect.py`). Add a config under
  `configs/ingest/` pointing at the raw directory (`source.type: batadal_dir`).
  Detection sniffs delimiter/header, finds the timestamp column (ISO or epoch
  s/ms/us/ns by magnitude), treats numeric columns as tags and names matching
  `/attack|label/i` as label columns. `source.overrides` can force any of these.
- **Single/multi CSV with known layout** → the generic CSV loader
  (`ingest/generic_csv.py`, `source.type: csv`): configure `time_column`,
  `tag_groups`, `label_column`, optional `resample`+`resample_agg`, and `missing`
  strategy. See `configs/ingest/swat_wadi_template.yaml`.
- **Already-tidy parquet** (DatetimeIndex + tag columns + label column) → point a
  `prepare` config's `source.path` straight at it (see
  `configs/prepare/example.yaml`).

## 2. Document units and validate

Add the dataset's tags to `configs/tags.yaml` (`unit`, optional `min`/`max`).
The validator (`ingest/validator.py`) then reports unit-less tags (error when
`require_units: true`), duplicates, gaps vs the expected step, flatline runs per
tag, and missing values. The ingest manifest records source file sha256s, row
counts, date ranges, detected schemas, and the quality report.

## 3. Labels

Either a sample-level label column (runs are derived into intervals by
`labeling.derive_intervals`; sentinel raw values go in `source.label_sentinels`
and become MISSING, never intervals) or an explicit intervals parquet
(`event_id`/`start`/`end`/`label`). Fill `label_column`/`label_map` in the
prepare config after inspecting the unique label values the ingest printed.

## 4. Windows → eval

`prepare` (window/label/split config) → `eval` (run config with `windows_path` or
inline `prepare:`) → `report`. Splits stay time/event-based; the leakage checker
runs in `prepare`.

## Tests to add

- Loader tests on a small synthetic copy of the real file layout (like
  `tests/test_ingest_detect_batadal.py` / `test_ingest_generic_csv.py`).
- Validator tests for the failure modes you care about (gaps, duplicates,
  unit-less tags) — see `tests/test_ingest_validator.py`.
- A missing-file test asserting the error message names the exact path to create.
