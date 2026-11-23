"""Generic CSV loader for SWaT/WADI-style files (access-gated; added later).

Everything is configurable — nothing is assumed: time column name, optional
time format, delimiter (sniffed if omitted), tag groups (or all numeric
non-label columns), label column. Handles sampling-rate differences via
optional resampling to a uniform frequency, and missing values via
interpolate/ffill/bfill/drop.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..data.loader import DataError
from .detect import parse_timestamp_column, sniff_delimiter

AGGS = ("mean", "median", "first", "last")
MISSING_STRATEGIES = ("interpolate", "ffill", "bfill", "drop")


def load_generic_csv(src: dict) -> tuple[pd.DataFrame, dict]:
    """Load one CSV per the ingest config's `source` section.

    Returns (tidy DataFrame, info dict). Tidy = DatetimeIndex + float tag
    columns + optional label column + source_file column.
    """
    p = Path(src["path"])
    if not p.is_file():
        raise DataError(
            f"CSV file not found: {p}. Place the SWaT/WADI file at this path or update "
            "source.path in the ingest config."
        )
    with open(p, "r", errors="replace") as fh:
        sample_text = "".join(fh.readline() for _ in range(200))
    delimiter = src.get("delimiter") or sniff_delimiter(sample_text)
    df = pd.read_csv(p, sep=delimiter, keep_default_na=False)
    df.columns = [str(c) for c in df.columns]

    time_col = src.get("time_column")
    if not time_col:
        raise DataError("source.time_column is required for source.type=csv")
    if time_col not in df.columns:
        raise DataError(
            f"{p}: time column {time_col!r} not found. Columns present: {list(df.columns)}. "
            "Update source.time_column in the ingest config."
        )

    label_col = src.get("label_column")
    if label_col and label_col not in df.columns:
        raise DataError(
            f"{p}: label column {label_col!r} not found. Columns present: {list(df.columns)}."
        )

    groups = src.get("tag_groups") or {}
    if groups:
        tags: list[str] = []
        for gname, cols in groups.items():
            if not isinstance(cols, list) or not cols:
                raise DataError(f"source.tag_groups.{gname} must be a non-empty list of columns")
            missing = [c for c in cols if c not in df.columns]
            if missing:
                raise DataError(
                    f"{p}: tag columns {missing} (group {gname!r}) not found. "
                    f"Columns present: {list(df.columns)}."
                )
            tags.extend(cols)
    else:
        tags = [
            c for c in df.columns
            if c not in (time_col, label_col)
            and pd.to_numeric(df[c].replace("", pd.NA), errors="coerce").notna().mean() >= 0.9
        ]
    if not tags:
        raise DataError(f"{p}: no tag columns selected; check source.tag_groups. Columns present: {list(df.columns)}")

    ts, _method = parse_timestamp_column(df[time_col])
    if src.get("time_format"):
        ts = pd.to_datetime(df[time_col], format=src["time_format"], errors="coerce")
    n_nat = int(ts.isna().sum())
    info = {
        "file": str(p),
        "delimiter": delimiter,
        "n_rows_raw": int(len(df)),
        "n_unparseable_timestamps": n_nat,
        "time_column": time_col,
        "label_column": label_col,
        "tag_columns": tags,
        "tag_groups": {k: list(v) for k, v in groups.items()} if groups else None,
    }
    df = df.loc[ts.notna()].copy()
    df.index = pd.DatetimeIndex(ts.dropna(), name=time_col)
    for c in tags:
        df[c] = pd.to_numeric(df[c].replace("", pd.NA), errors="coerce")
    if label_col:
        df[label_col] = df[label_col].replace("", pd.NA)
    out = df[tags + ([label_col] if label_col else [])].sort_index()

    resample = src.get("resample")
    if resample:
        freq = pd.Timedelta(resample)
        if freq <= pd.Timedelta(0):
            raise DataError("source.resample must be a positive duration, e.g. '1s', '5min'")
        agg = src.get("resample_agg", "mean")
        if agg not in AGGS:
            raise DataError(f"source.resample_agg must be one of {AGGS}, got {agg!r}")
        resampled = out[tags].resample(freq).agg(agg)
        if label_col:
            labels = out[label_col].reindex(resampled.index, method="ffill")
            fill = src.get("label_fill", "normal")
            labels = labels.fillna(fill)
            resampled[label_col] = labels
        info["resample"] = resample
        info["resample_agg"] = agg
        info["n_rows_resampled"] = int(len(resampled))
        out = resampled

    missing = src.get("missing", "interpolate")
    if missing not in MISSING_STRATEGIES:
        raise DataError(f"source.missing must be one of {MISSING_STRATEGIES}, got {missing!r}")
    limit = src.get("missing_limit")
    if missing == "interpolate":
        out[tags] = out[tags].interpolate(method="time", limit=limit)
    elif missing == "ffill":
        out[tags] = out[tags].ffill(limit=limit)
    elif missing == "bfill":
        out[tags] = out[tags].bfill(limit=limit)
    else:
        out = out.dropna(subset=tags)
    info["missing_strategy"] = missing
    info["n_missing_after"] = {c: int(out[c].isna().sum()) for c in tags}
    info["n_rows_out"] = int(len(out))
    out = out.copy()
    out["source_file"] = p.name
    return out, info
