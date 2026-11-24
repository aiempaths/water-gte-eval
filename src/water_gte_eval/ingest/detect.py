"""Schema detection for raw files. Nothing is assumed: delimiter, header,
timestamp column, tag columns, and label columns are detected from the file
itself (and can be overridden from the ingest config)."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

SUPPORTED_SUFFIXES = {".csv", ".dat", ".txt", ".tsv"}
LABEL_NAME_RE = re.compile(r"(attack|label)", re.IGNORECASE)


class DetectError(Exception):
    pass


@dataclass
class DetectedSchema:
    file: str
    delimiter: str
    has_header: bool
    columns: list[str]
    timestamp_column: str | None
    timestamp_method: str | None  # "iso" | "epoch_s" | "epoch_ms" | ...
    tag_columns: list[str]
    label_columns: list[str]
    n_rows: int
    label_value_counts: dict = field(default_factory=dict)  # label col -> {value: count} (capped)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "file": self.file,
            "delimiter": self.delimiter,
            "has_header": self.has_header,
            "columns": self.columns,
            "timestamp_column": self.timestamp_column,
            "timestamp_method": self.timestamp_method,
            "tag_columns": self.tag_columns,
            "label_columns": self.label_columns,
            "n_rows": self.n_rows,
            "label_value_counts": {k: dict(v) for k, v in self.label_value_counts.items()},
            "notes": list(self.notes),
        }


def sniff_delimiter(sample_text: str) -> str:
    try:
        return csv.Sniffer().sniff(sample_text, delimiters=",;\t|").delimiter
    except csv.Error:
        counts = {d: sample_text.count(d) for d in (",", ";", "\t", "|")}
        best = max(counts, key=counts.get)
        return best if counts[best] > 0 else ","


def sniff_header(sample_text: str, delimiter: str) -> bool:
    try:
        if csv.Sniffer().has_header(sample_text):
            return True
    except csv.Error:
        pass
    rows = list(csv.reader(sample_text.splitlines()[:20], delimiter=delimiter))
    rows = [r for r in rows if r]
    if len(rows) < 2:
        return False
    first = rows[0]
    numeric_first = sum(1 for c in first if _is_float(c))
    numeric_second = sum(1 for c in rows[1] if _is_float(c))
    return numeric_first < len(first) and numeric_second >= max(1, len(rows[1]) // 2)


def _is_float(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def _epoch_unit(values: pd.Series) -> str | None:
    v = values.astype("float64")
    if len(v) == 0 or ((v > 1e8) & (v < 1e19)).mean() < 0.9:
        return None
    med = float(v.median())
    if med < 1e11:
        return "s"
    if med < 1e14:
        return "ms"
    if med < 1e17:
        return "us"
    return "ns"


def _empty(idx) -> pd.Series:
    return pd.Series(pd.NaT, index=idx, dtype="datetime64[ns]")


def parse_timestamp_sample(sample: pd.Series) -> tuple[pd.Series, str | None]:
    """Try to interpret a column sample as timestamps. Returns (parsed, method).

    Numeric-looking columns are only accepted as epoch seconds/ms/us/ns (magnitude
    1e8..1e19) — small integers must NOT be misread as nanosecond timestamps.
    """
    s = sample.dropna()
    if s.empty:
        return _empty(s.index), None
    num = pd.to_numeric(s, errors="coerce")
    if len(num) and num.notna().mean() >= 0.9:
        unit = _epoch_unit(num.dropna())
        if unit:
            return pd.to_datetime(num, unit=unit), f"epoch_{unit}"
        return _empty(s.index), None
    for kwargs in ({}, {"format": "mixed"}):
        try:
            parsed = pd.to_datetime(s, errors="coerce", **kwargs)
        except (ValueError, TypeError):
            continue
        if len(parsed) and parsed.notna().mean() >= 0.8:
            return parsed, "iso"
    return _empty(s.index), None


def parse_timestamp_column(s: pd.Series) -> tuple[pd.Series, str | None]:
    """Parse a full column to datetimes (NaT where unparseable)."""
    num = pd.to_numeric(s, errors="coerce")
    if len(num) and num.notna().mean() >= 0.9:
        unit = _epoch_unit(num.dropna())
        if unit:
            return pd.to_datetime(num, unit=unit, errors="coerce"), f"epoch_{unit}"
        return _empty(s.index), None
    for kwargs in ({}, {"format": "mixed"}):
        try:
            parsed = pd.to_datetime(s, errors="coerce", **kwargs)
        except (ValueError, TypeError):
            continue
        if parsed.notna().mean() >= 0.8:
            return parsed, "iso"
    return _empty(s.index), None


def detect_schema(path: str | Path, sample_rows: int = 500) -> DetectedSchema:
    p = Path(path)
    if not p.is_file():
        raise DetectError(f"not a file: {p}")
    with open(p, "r", errors="replace") as fh:
        sample_text = "".join(fh.readline() for _ in range(200))
    delimiter = sniff_delimiter(sample_text)
    has_header = sniff_header(sample_text, delimiter)
    df = pd.read_csv(p, sep=delimiter, header=0 if has_header else None,
                     dtype=str, keep_default_na=False)
    if not has_header:
        df.columns = [f"col_{i}" for i in range(df.shape[1])]
    columns = [str(c) for c in df.columns]
    head = df.head(sample_rows)

    ts_col: str | None = None
    ts_method: str | None = None
    best = -1.0
    for c in columns:
        parsed, method = parse_timestamp_sample(head[c])
        rate = float(parsed.notna().mean()) if len(parsed) else 0.0
        if method and rate >= 0.8 and rate > best:
            best, ts_col, ts_method = rate, c, method

    label_columns = [c for c in columns if c != ts_col and LABEL_NAME_RE.search(c)]
    tag_columns: list[str] = []
    notes: list[str] = []
    for c in columns:
        if c == ts_col or c in label_columns:
            continue
        num = pd.to_numeric(head[c].replace("", pd.NA), errors="coerce")
        if len(num) and num.notna().mean() >= 0.9:
            tag_columns.append(c)
        else:
            notes.append(f"column {c!r} is not numeric and not a label; excluded from tags")
    if ts_col is None:
        notes.append(
            "no timestamp column detected (datetime-string or epoch); "
            "set source.overrides.timestamp_column in the ingest config"
        )
    label_value_counts = {}
    for c in label_columns:
        vc = df[c].astype(str).value_counts()
        label_value_counts[c] = {str(k): int(v) for k, v in vc.head(50).items()}
        if len(vc) > 50:
            notes.append(f"label column {c!r} has {len(vc)} unique values; showing top 50")
    return DetectedSchema(
        file=p.name,
        delimiter=delimiter,
        has_header=has_header,
        columns=columns,
        timestamp_column=ts_col,
        timestamp_method=ts_method,
        tag_columns=tag_columns,
        label_columns=label_columns,
        n_rows=int(len(df)),
        label_value_counts=label_value_counts,
        notes=notes,
    )


def list_supported_files(directory: str | Path) -> list[Path]:
    d = Path(directory)
    if not d.is_dir():
        return []
    return sorted(p for p in d.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES)
