"""Ingest config: one YAML per ingest run (separate from experiment configs)."""

from __future__ import annotations

from pathlib import Path

import yaml

from ..config import config_hash

SOURCE_TYPES = ("batadal_dir", "csv")
AGGS = ("mean", "median", "first", "last")
MISSING_STRATEGIES = ("interpolate", "ffill", "bfill", "drop")


class IngestConfigError(Exception):
    pass


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise IngestConfigError(msg)


def load_ingest_config(path: str | Path) -> dict:
    p = Path(path)
    if not p.is_file():
        raise IngestConfigError(f"ingest config not found: {p}")
    try:
        raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise IngestConfigError(f"{p} is not valid YAML: {exc}") from exc
    _check(isinstance(raw, dict), "ingest config root must be a mapping")
    _check(isinstance(raw.get("name"), str) and raw["name"], "'name' must be a non-empty string")

    src = raw.get("source")
    _check(isinstance(src, dict), "'source' must be a mapping")
    _check(src.get("type") in SOURCE_TYPES, f"source.type must be one of {SOURCE_TYPES}")
    _check(isinstance(src.get("path"), str) and src["path"], "source.path must be a non-empty string")
    if src.get("files") is not None:
        _check(isinstance(src["files"], list) and all(isinstance(x, str) for x in src["files"]),
               "source.files must be a list of file names")
    if src.get("overrides") is not None:
        _check(isinstance(src["overrides"], dict), "source.overrides must be a mapping")
    if src["type"] == "csv":
        _check(isinstance(src.get("time_column"), str) and src["time_column"],
               "source.time_column is required for source.type=csv")
        if src.get("tag_groups") is not None:
            _check(isinstance(src["tag_groups"], dict)
                   and all(isinstance(v, list) for v in src["tag_groups"].values()),
                   "source.tag_groups must map group name -> list of columns")
        _check(src.get("resample_agg", "mean") in AGGS, f"source.resample_agg must be one of {AGGS}")
        _check(src.get("missing", "interpolate") in MISSING_STRATEGIES,
               f"source.missing must be one of {MISSING_STRATEGIES}")
        if src.get("missing_limit") is not None:
            _check(isinstance(src["missing_limit"], int), "source.missing_limit must be an int or null")

    out = raw.get("output")
    _check(isinstance(out, dict) and isinstance(out.get("dir"), str) and out["dir"],
           "output.dir must be a non-empty string")

    if raw.get("tags_yaml") is not None:
        _check(isinstance(raw["tags_yaml"], str), "tags_yaml must be a path string or null")
    if raw.get("expected_step") is not None:
        _check(isinstance(raw["expected_step"], str), "expected_step must be a duration string or null")
    _check(isinstance(raw.get("flatline_min_run", 10), int), "flatline_min_run must be an int")
    _check(isinstance(raw.get("require_units", False), bool), "require_units must be a bool")

    raw["_path"] = str(p)
    raw["_config_hash"] = config_hash({k: v for k, v in raw.items() if not k.startswith("_")})
    return raw
