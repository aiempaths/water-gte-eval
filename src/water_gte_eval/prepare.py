"""Prepare pipeline: processed parquet -> labeled sample windows -> staged split.

Reads a tidy parquet from ingest (DatetimeIndex, one column per tag, optional
sample-level label column), builds sample-based sliding windows, labels them
from labeled intervals (explicit file, or derived from label-column runs) with
the configurable overlap rule + lag buffer, assigns time-ordered train/val/test
stages (events never split across stages, no cross-stage time overlap), runs the
leakage checker, and writes:
  <name>__<stem>_windows.parquet    one row per window (stage, label, event_id, per-tag arrays)
  <name>__<stem>_intervals.parquet  the labeled intervals with event_id
  prepare.json                      counts per split/label, leakage report, hashes
Prints the split summary (window counts per split and per label).
Deterministic: no wall-clock fields anywhere.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from .config import config_hash
from .data import (
    TelemetryColumns,
    assign_stages,
    build_sample_windows,
    check_leakage,
    derive_intervals,
    file_sha256,
    format_split_summary,
    label_windows,
    load_table,
    require_columns,
    split_summary,
)


class PrepareConfigError(Exception):
    pass


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise PrepareConfigError(msg)


def load_prepare_config(path: str | Path) -> dict:
    p = Path(path)
    if not p.is_file():
        raise PrepareConfigError(f"prepare config not found: {p}")
    try:
        raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise PrepareConfigError(f"{p} is not valid YAML: {exc}") from exc
    _check(isinstance(raw, dict), "prepare config root must be a mapping")
    _check(isinstance(raw.get("name"), str) and raw["name"], "'name' must be a non-empty string")
    src = raw.get("source")
    _check(isinstance(src, dict) and isinstance(src.get("path"), str) and src["path"],
           "source.path must be a non-empty string (parquet file or directory)")
    if src.get("tag_columns") is not None:
        _check(isinstance(src["tag_columns"], list), "source.tag_columns must be a list or null")
    if src.get("label_sentinels") is not None:
        _check(isinstance(src["label_sentinels"], list),
               "source.label_sentinels must be a list of raw label values treated as MISSING")
    win = raw.get("window")
    _check(isinstance(win, dict), "'window' must be a mapping")
    _check(isinstance(win.get("length"), int) and win["length"] >= 1, "window.length must be an int >= 1")
    _check(isinstance(win.get("stride"), int) and win["stride"] >= 1, "window.stride must be an int >= 1")
    lab = raw.get("labeling", {})
    _check(isinstance(lab, dict), "'labeling' must be a mapping")
    _check(0 < float(lab.get("min_overlap_frac", 0.5)) <= 1, "labeling.min_overlap_frac must be in (0, 1]")
    sp = raw.get("split")
    _check(isinstance(sp, dict), "'split' must be a mapping")
    vf, tf = float(sp.get("val_frac", 0.15)), float(sp.get("test_frac", 0.15))
    _check(0 < vf < 1 and 0 < tf < 1 and vf + tf < 1, "split needs 0 < val_frac, 0 < test_frac, sum < 1")
    out = raw.get("output")
    _check(isinstance(out, dict) and isinstance(out.get("dir"), str) and out["dir"],
           "output.dir must be a non-empty string")
    raw["_path"] = str(p)
    raw["_config_hash"] = config_hash({k: v for k, v in raw.items() if not k.startswith("_")})
    return raw


def _resolve(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else Path.cwd() / p


def _dump(obj: dict) -> str:
    return json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n"


def _load_intervals(cfg: dict) -> pd.DataFrame | None:
    iv = cfg.get("intervals")
    if not iv:
        return None
    p = _resolve(iv["path"])
    df = load_table(p, iv.get("format"))
    cols = iv.get("columns") or {}
    need = {"event_id": "event_id", "start": "start", "end": "end", "label": "label"}
    mapped = {k: cols.get(k, v) for k, v in need.items()}
    require_columns(df, list(mapped.values()), f"Intervals file {p}")
    out = pd.DataFrame({
        "event_id": df[mapped["event_id"]].astype(str),
        "start": pd.to_datetime(df[mapped["start"]]),
        "end": pd.to_datetime(df[mapped["end"]]),
        "label": df[mapped["label"]].astype(str),
    })
    return out


def _prepare_one(df: pd.DataFrame, cfg: dict, intervals: pd.DataFrame | None, src_file: str) -> dict:
    src = cfg["source"]
    label_col = src.get("label_column")
    tags = src.get("tag_columns")
    if tags is None:
        skip = {label_col, "source_file"} - {None}
        tags = [
            c for c in df.columns
            if c not in skip
            and c != df.index.name
            and pd.api.types.is_numeric_dtype(df[c])
        ]
    if not tags:
        raise PrepareConfigError(
            f"{src_file}: no tag columns selected. Columns: {list(df.columns)}"
        )
    tcols = TelemetryColumns(timestamp=df.index.name or "timestamp", sensors=tuple(tags))

    if intervals is None:
        if not label_col:
            raise PrepareConfigError(
                "either source.label_column (to derive intervals from label runs) or "
                "intervals.path must be set"
            )
        if label_col not in df.columns:
            raise PrepareConfigError(
                f"{src_file}: label column {label_col!r} not found. Columns present: {list(df.columns)}. "
                "Set source.label_column in the prepare config."
            )
        intervals = derive_intervals(
            df, label_col,
            label_map=src.get("label_map"),
            default_label=src.get("default_label", "normal"),
            sentinels=src.get("label_sentinels"),
        )

    lab = cfg.get("labeling", {})
    windows = build_sample_windows(
        df, tcols, length=cfg["window"]["length"], stride=cfg["window"]["stride"],
        tag_group=src.get("tag_group", "default"),
    )
    label_windows(
        windows, intervals,
        min_overlap_frac=float(lab.get("min_overlap_frac", 0.5)),
        label_overrides=lab.get("label_overrides"),
        lag=pd.Timedelta(lab.get("lag", "0s")),
        priority=lab.get("priority"),
        default_label=src.get("default_label", "normal"),
    )
    sp = cfg["split"]
    split = assign_stages(windows, val_frac=float(sp["val_frac"]), test_frac=float(sp["test_frac"]))
    leakage = check_leakage(split.train, split.val, split.test)  # raises LeakageError if leaky
    summary = split_summary(split)
    return {
        "windows": windows,
        "intervals": intervals,
        "summary": summary,
        "leakage": leakage,
        "tags": tags,
        "src_file": src_file,
    }


def run_prepare(config_path: str | Path) -> Path:
    cfg = load_prepare_config(config_path)
    out_dir = _resolve(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    src_path = _resolve(cfg["source"]["path"])
    if src_path.is_dir():
        files = sorted(p for p in src_path.glob("*.parquet")
                       if not p.name.endswith(("_windows.parquet", "_intervals.parquet")))
        if not files:
            raise PrepareConfigError(
                f"no parquet files in {src_path}. Run ingest first (outputs go to the "
                "ingest output.dir, e.g. data/processed/)."
            )
    elif src_path.is_file():
        files = [src_path]
    else:
        raise PrepareConfigError(
            f"source.path not found: {src_path}. Run ingest first or fix the path."
        )

    intervals = _load_intervals(cfg)
    outputs: dict = {}
    for f in files:
        df = pd.read_parquet(f)
        if not isinstance(df.index, pd.DatetimeIndex):
            raise PrepareConfigError(
                f"{f}: expected a tidy parquet with a DatetimeIndex (as written by ingest); "
                f"index type is {type(df.index).__name__}"
            )
        res = _prepare_one(df, cfg, intervals, f.name)
        out_name = f"{cfg['name']}__{f.stem}"
        wdf = pd.DataFrame({
            "window_id": [w.window_id for w in res["windows"]],
            "start": [w.start for w in res["windows"]],
            "end": [w.end for w in res["windows"]],
            "stage": [w.stage if w.stage is not None else "dropped" for w in res["windows"]],
            "label": [w.label for w in res["windows"]],
            "event_id": [w.event_id if w.event_id is not None else "" for w in res["windows"]],
            "tag_group": [w.tag_group for w in res["windows"]],
            "source_file": [res["src_file"]] * len(res["windows"]),
            **{t: [a.tolist() for a in (w.sensor_arrays()[t] for w in res["windows"])] for t in res["tags"]},
        })
        wpath = out_dir / f"{out_name}_windows.parquet"
        wdf.to_parquet(wpath, index=False)
        ipath = out_dir / f"{out_name}_intervals.parquet"
        res["intervals"].to_parquet(ipath, index=False)
        print(format_split_summary(res["summary"]))
        print(f"  leakage check: ok (no time overlap, no shared event_id across stages)")
        outputs[out_name] = {
            "windows_file": wpath.name,
            "windows_sha256": file_sha256(wpath),
            "intervals_file": ipath.name,
            "intervals_sha256": file_sha256(ipath),
            "n_windows": len(res["windows"]),
            "n_windows_staged": sum(s["n"] for s in res["summary"]["stages"].values()),
            "n_guard_band_dropped": len(res["windows"]) - sum(
                s["n"] for s in res["summary"]["stages"].values()),
            "n_intervals": int(len(res["intervals"])),
            "window": {"length": cfg["window"]["length"], "stride": cfg["window"]["stride"],
                       "tag_group": cfg["source"].get("tag_group", "default")},
            "split": res["summary"],
            "leakage": res["leakage"],
            "source": {"file": str(f), "sha256": file_sha256(f)},
        }

    manifest = {"name": cfg["name"], "config_hash": cfg["_config_hash"], "outputs": outputs}
    mpath = out_dir / "prepare.json"
    mpath.write_text(_dump(manifest), encoding="utf-8")
    print(f"prepare manifest: {mpath}")
    return mpath
