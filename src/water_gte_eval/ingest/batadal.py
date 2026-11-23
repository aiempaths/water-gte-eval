"""BATADAL (C-Town) loader: schema-detecting, no assumed file or column names.

Every tabular file (.csv/.dat/.txt/.tsv) in the source directory is sniffed:
delimiter, header, timestamp column (ISO strings or epoch s/ms/us/ns), numeric
tag columns, and label columns (names matching /attack|label/i). Detection can
be overridden per run via source.overrides in the ingest config.

Returns tidy DataFrames: DatetimeIndex + one float column per tag + the attack
label column(s) + a source_file column when loading a directory.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..data.loader import DataError
from .detect import DetectedSchema, detect_schema, list_supported_files, parse_timestamp_column

BATADAL_MISSING_MSG = (
    "BATADAL raw files not found: {dir} does not exist or contains no "
    ".csv/.dat/.txt/.tsv files. Place the raw BATADAL (C-Town) files you "
    "downloaded into {dir} — one file per scenario is fine; the loader "
    "auto-detects each file's delimiter, header, timestamp column, tag columns, "
    "and attack-label columns (no names are hard-coded). If labels live in a "
    "separate file, put it in the same directory; override detection with "
    "source.overrides (timestamp_column / label_columns / tag_columns) in the "
    "ingest config if needed."
)


def load_batadal_file(
    path: str | Path,
    overrides: dict | None = None,
    add_source_column: bool = False,
) -> tuple[pd.DataFrame, DetectedSchema]:
    p = Path(path)
    schema = detect_schema(p)
    ov = overrides or {}
    if ov.get("timestamp_column"):
        if ov["timestamp_column"] not in schema.columns:
            raise DataError(
                f"{p}: overrides.timestamp_column {ov['timestamp_column']!r} not in file; "
                f"columns present: {schema.columns}"
            )
        schema.timestamp_column = ov["timestamp_column"]
    if ov.get("label_columns"):
        missing = [c for c in ov["label_columns"] if c not in schema.columns]
        if missing:
            raise DataError(f"{p}: overrides.label_columns {missing} not in file; columns present: {schema.columns}")
        schema.label_columns = list(ov["label_columns"])
    if ov.get("tag_columns"):
        missing = [c for c in ov["tag_columns"] if c not in schema.columns]
        if missing:
            raise DataError(f"{p}: overrides.tag_columns {missing} not in file; columns present: {schema.columns}")
        schema.tag_columns = list(ov["tag_columns"])

    if schema.timestamp_column is None:
        raise DataError(
            f"{p}: could not detect a timestamp column (tried ISO datetime parsing and "
            f"epoch s/ms/us/ns detection). Columns found: {schema.columns}. "
            "Set source.overrides.timestamp_column in the ingest config."
        )

    df = pd.read_csv(p, sep=schema.delimiter, header=0 if schema.has_header else None,
                     keep_default_na=False)
    if not schema.has_header:
        df.columns = [f"col_{i}" for i in range(df.shape[1])]
    df.columns = [str(c) for c in df.columns]

    ts_parsed, method = parse_timestamp_column(df[schema.timestamp_column])
    n_nat = int(ts_parsed.isna().sum())
    if n_nat:
        schema.notes.append(f"{n_nat} unparseable timestamps in {schema.timestamp_column!r} -> NaT (dropped)")
    df = df.loc[ts_parsed.notna()].copy()
    df.index = pd.DatetimeIndex(ts_parsed.dropna(), name=schema.timestamp_column)

    out_cols: list[str] = []
    for c in schema.tag_columns:
        df[c] = pd.to_numeric(df[c].replace("", pd.NA), errors="coerce").astype("float64")
        out_cols.append(c)
    for c in schema.label_columns:
        df[c] = df[c].replace("", pd.NA)
        out_cols.append(c)
    out = df[out_cols].sort_index()
    if add_source_column:
        out["source_file"] = p.name
    schema.timestamp_method = method
    return out, schema


def load_batadal_dir(
    directory: str | Path,
    files: list[str] | None = None,
    overrides: dict | None = None,
) -> dict[str, tuple[pd.DataFrame, DetectedSchema]]:
    d = Path(directory)
    found = list_supported_files(d)
    if not found:
        raise DataError(BATADAL_MISSING_MSG.format(dir=d))
    if files:
        wanted = set(files)
        missing = sorted(wanted - {p.name for p in found})
        if missing:
            raise DataError(
                f"{d}: source.files {missing} not found; available: {[p.name for p in found]}"
            )
        found = [p for p in found if p.name in wanted]
    return {p.name: load_batadal_file(p, overrides=overrides, add_source_column=True) for p in found}
