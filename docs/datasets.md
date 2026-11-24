# Dataset placement and expectations

The harness **never invents** dataset URLs, column names, or formats. It inspects
whatever files are present and stops with an exact message when something is missing.

## BATADAL (C-Town water distribution benchmark)

Place the raw files in:

```
data/raw/batadal/          # every .csv/.dat/.txt/.tsv file in this directory is read
```

Then:

```bash
python -m water_gte_eval ingest configs/ingest/batadal.yaml
```

The BATADAL directory loader (`src/water_gte_eval/ingest/batadal.py` +
`ingest/detect.py`) **schema-detects** each file: delimiter and header are sniffed;
the timestamp column is found by parsing (ISO strings, or epoch s/ms/us/ns chosen by
magnitude — small integers are not misread as nanoseconds); numeric columns become
tags; columns whose names match `/attack|label/i` become label columns. Nothing is
hard-coded; `source.overrides` in `configs/ingest/batadal.yaml` can override any
detected field. When the run starts, the loader prints the detected schema per file
and the **unique values of every label column** (also stored as
`label_value_counts` in the ingest manifest) — use that output to fill
`label_column` / `label_map` and any sentinel values (`source.label_sentinels`, e.g.
`-1`/`-999`/empty → MISSING, never an interval) in `configs/prepare/batadal.yaml`.

If `data/raw/batadal/` does not exist, ingest stops with the exact path to create.

Processed output: `data/processed/*.parquet` (tidy: DatetimeIndex + one float column
per tag + label columns + `source_file`) plus `data/processed/manifest.json`
(source file sha256s, row counts, date ranges, detected schemas, quality report).

## SWaT / WADI (access-gated)

Place the CSV export under:

```
data/raw/swat/    # or data/raw/wadi/
```

Then copy `configs/ingest/swat_wadi_template.yaml`, set `source.path` and the REAL
column names (the template has placeholders and refuses to guess), and run
`python -m water_gte_eval ingest <your config>`. The generic CSV loader
(`ingest/generic_csv.py`) supports `time_column`, `tag_groups`, `label_column`,
delimiter sniffing, `resample`+`resample_agg` to unify sampling rates, and
`missing` strategies (interpolate/ffill/bfill/drop with optional `missing_limit`).

## Data quality validation

`ingest/validator.py` checks, per file: monotonic time, duplicate timestamps, gaps
vs the expected step (inferred when unset), flatline (constant) runs per tag,
missing values, and unit-less tags vs `configs/tags.yaml` (`require_units: true`
makes unit-less tags an error; optional min/max range checks). The report is printed
and stored in the manifest. Tests cover gaps, duplicates, and unit-less tags.

## Synthetic examples (NOT real plant data)

```bash
python -m water_gte_eval make-example-raw --out examples/raw          # raw csvs for ingest
python -m water_gte_eval make-example-processed --out examples/processed_demo  # tidy parquet for prepare
python -m water_gte_eval make-example-data --out examples/data        # legacy `run` demo
python -m water_gte_eval sim-wntr configs/sim/wntr_example.yaml       # WNTR on bundled EPANET Net3
python -m water_gte_eval sim-toy configs/sim/toy_example.yaml         # labeled TOY plant model
```
