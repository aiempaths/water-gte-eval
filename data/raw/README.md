# Raw data placement

- `data/raw/batadal/` — place the raw BATADAL (C-Town) files here (the tabular
  `.csv` / `.dat` / `.txt` / `.tsv` files from the distribution, one per
  scenario is fine). Then run:
  `python -m water_gte_eval ingest configs/ingest/batadal.yaml`
  The loader auto-detects each file's delimiter, header, timestamp column, tag
  columns, and attack-label columns — no file or column names are hard-coded.
  If labels live in a separate file, place it in the same directory.
- `data/raw/swat/` (or `data/raw/wadi/`) — place the access-gated SWaT/WADI CSV
  here and fill in the real column names in `configs/ingest/swat_wadi_template.yaml`.

Processed output (parquet + manifest.json with source hashes, row counts, date
ranges, and the data-quality report) goes to `data/processed/`.
