"""Ingest pipeline: raw files -> tidy parquet in the output dir + manifest.json.

Deterministic: the manifest contains only source file hashes, row counts, date
ranges, detected schemas, and quality reports — no wall-clock timestamps — so
re-running the same ingest config reproduces identical outputs. The data-quality
report is printed to stdout.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..data.loader import file_sha256
from .batadal import load_batadal_dir
from .config import load_ingest_config
from .generic_csv import load_generic_csv
from .validator import format_quality_report, load_tags_yaml, validate_quality


def _resolve(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else Path.cwd() / p


def _dump(obj: dict) -> str:
    return json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n"


def run_ingest(config_path: str | Path) -> Path:
    cfg = load_ingest_config(config_path)
    src = cfg["source"]
    out_dir = _resolve(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    tags_cfg = load_tags_yaml(_resolve(cfg["tags_yaml"]) if cfg.get("tags_yaml") else None)
    expected_step = pd.Timedelta(cfg["expected_step"]) if cfg.get("expected_step") else None
    flatline_min_run = int(cfg.get("flatline_min_run", 10))
    require_units = bool(cfg.get("require_units", False))

    frames: dict[str, tuple[pd.DataFrame, dict]] = {}
    if src["type"] == "batadal_dir":
        loaded = load_batadal_dir(_resolve(src["path"]), files=src.get("files"),
                                  overrides=src.get("overrides"))
        for fname, (df, schema) in loaded.items():
            stem = Path(fname).stem
            out_name = f"{cfg['name']}__{stem}"
            frames[out_name] = (df, {
                "source_file": fname,
                "source_sha256": file_sha256(_resolve(src["path"]) / fname),
                "schema": schema.to_dict(),
                "tag_columns": schema.tag_columns,
                "label_columns": schema.label_columns,
                "timestamp_column": schema.timestamp_column,
            })
    else:
        df, info = load_generic_csv({**src, "path": str(_resolve(src["path"]))})
        out_name = cfg["name"]
        frames[out_name] = (df, {
            "source_file": info["file"],
            "source_sha256": file_sha256(info["file"]),
            "schema": info,
            "tag_columns": info["tag_columns"],
            "label_columns": [info["label_column"]] if info["label_column"] else [],
            "timestamp_column": info["time_column"],
        })

    outputs: dict = {}
    quality: dict = {}
    for out_name, (df, meta) in sorted(frames.items()):
        label_cols = [c for c in meta["label_columns"] if c in df.columns]
        tag_cols = [c for c in meta["tag_columns"] if c in df.columns]
        report = validate_quality(
            df, tag_cols, tags_cfg=tags_cfg, expected_step=expected_step,
            flatline_min_run=flatline_min_run, require_units=require_units,
            label_columns=label_cols,
        )
        print(format_quality_report(out_name, report))
        parquet_name = f"{out_name}.parquet"
        df.to_parquet(out_dir / parquet_name)
        outputs[out_name] = {
            "file": parquet_name,
            "sha256": file_sha256(out_dir / parquet_name),
            "n_rows": int(len(df)),
            "time_min": df.index.min().isoformat() if len(df) else None,
            "time_max": df.index.max().isoformat() if len(df) else None,
            "tag_columns": tag_cols,
            "label_columns": label_cols,
            "timestamp_column": meta["timestamp_column"],
        }
        quality[out_name] = report
        outputs[out_name]["source"] = {"file": meta["source_file"], "sha256": meta["source_sha256"]}
        outputs[out_name]["detected_schema"] = meta["schema"]

    manifest_path = out_dir / "manifest.json"
    manifest = {"runs": {}}
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(manifest.get("runs"), dict):
                manifest = {"runs": {}}
        except json.JSONDecodeError:
            manifest = {"runs": {}}
    manifest["runs"][cfg["name"]] = {
        "config_hash": cfg["_config_hash"],
        "source_type": src["type"],
        "outputs": outputs,
        "quality": quality,
    }
    manifest_path.write_text(_dump(manifest), encoding="utf-8")
    print(f"manifest: {manifest_path}")
    return manifest_path
