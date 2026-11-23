"""Loading and validating real data files.

The harness never assumes URLs, column names, or formats: the YAML config
names the file paths and the column mapping, and the loader checks them
against the actual file on disk. If a file is missing it says exactly what to
place where; if a column is missing it lists the columns that do exist.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd


class DataError(Exception):
    """Raised when a data file is missing, unreadable, or lacks mapped columns."""


SUPPORTED_FORMATS = (".parquet", ".csv")


def file_sha256(path: str | Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def load_table(path: str | Path, fmt: str | None = None) -> pd.DataFrame:
    """Load a .parquet or .csv file. Format comes from config or the extension."""
    p = Path(path)
    if not p.is_file():
        raise DataError(
            f"Data file not found: {p}. Place the file at this path "
            f"(supported formats: {', '.join(SUPPORTED_FORMATS)}) or update 'path' in the YAML config."
        )
    ext = (fmt or p.suffix).lower().lstrip(".")
    if ext == "parquet":
        try:
            return pd.read_parquet(p)
        except Exception as exc:
            raise DataError(f"Could not read parquet file {p}: {exc}") from exc
    if ext == "csv":
        try:
            return pd.read_csv(p)
        except Exception as exc:
            raise DataError(f"Could not read csv file {p}: {exc}") from exc
    raise DataError(
        f"Unsupported format '{ext}' for {p}. Supported: {', '.join(SUPPORTED_FORMATS)}."
    )


def require_columns(df: pd.DataFrame, cols: list[str], file_desc: str) -> None:
    """Fail with a precise message if any mapped column is absent from the file."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise DataError(
            f"{file_desc} is missing mapped column(s) {missing}. "
            f"Columns present in the file: {sorted(str(c) for c in df.columns)}. "
            "Update the column mapping in the YAML config to the real column names."
        )
