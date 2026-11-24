"""Artifact-level tamper audit for a run directory.

verify_run_receipts() re-derives every engine receipt in
runs/<hash>/results.parquet from the run's own windows (config.yaml ->
windows_path or inline prepare) and the sha256 chain contract in
gte_eval.engine_adapter.base. Scoring order within a mode is
sorted by (start, window_id), exactly as engine_run._run_mode scored them.
A modified row fails and is identified by mode, chain position, and window_id.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from gte_eval.engine_adapter.base import GENESIS, compute_receipt

from .engine_run import _load_or_build_windows, load_run_config


def verify_run_receipts(run_dir: str | Path) -> dict:
    run_dir = Path(run_dir)
    results_path = run_dir / "results.parquet"
    config_path = run_dir / "config.yaml"
    if not results_path.is_file() or not config_path.is_file():
        raise FileNotFoundError(
            f"{run_dir} must contain results.parquet and config.yaml (a run dir from `eval`)"
        )
    cfg = load_run_config(config_path)
    # configs use repo-root-relative paths; resolve them against the CWD first,
    # then against the run dir's grandparent (runs/<hash> -> repo root)
    for key in ("windows_path",):
        if cfg.get(key) and not Path(cfg[key]).is_absolute():
            p = Path(cfg[key])
            if not p.exists() and (run_dir.parents[1] / p).exists():
                cfg[key] = str(run_dir.parents[1] / p)
    if isinstance(cfg.get("prepare"), dict) and isinstance(cfg["prepare"].get("source"), dict):
        sp = cfg["prepare"]["source"].get("path")
        if sp and not Path(sp).is_absolute():
            p = Path(sp)
            if not p.exists() and (run_dir.parents[1] / p).exists():
                cfg["prepare"]["source"]["path"] = str(run_dir.parents[1] / p)
    windows, _ = _load_or_build_windows(cfg)
    by_id = {w.window_id: w.to_adapter() for w in windows}

    res = pd.read_parquet(results_path)
    breaks: list[dict] = []
    n_checked = 0
    for mode in sorted(res["mode"].unique()):
        sub = res[res["mode"] == mode].sort_values(["start", "window_id"])
        prev = GENESIS
        for pos, (_, row) in enumerate(sub.iterrows()):
            aw = by_id.get(str(row["window_id"]))
            if aw is None:
                breaks.append({"mode": mode, "position": pos, "window_id": str(row["window_id"]),
                               "expected": None, "actual": str(row["engine_receipt"]),
                               "reason": "window_id not present in the run's windows"})
                prev = str(row["engine_receipt"])
                n_checked += 1
                continue
            outputs = {
                "closeness": float(row["engine_closeness"]),
                "agreement": float(row["engine_agreement"]),
                "verdict": str(row["engine_verdict"]),
                "neighbor_ids": [str(x) for x in row["engine_neighbors"]],
            }
            expected = compute_receipt(prev, aw, outputs)
            n_checked += 1
            if expected != str(row["engine_receipt"]):
                breaks.append({"mode": mode, "position": pos, "window_id": str(row["window_id"]),
                               "expected": expected, "actual": str(row["engine_receipt"])})
            prev = str(row["engine_receipt"])
    return {
        "run_dir": str(run_dir),
        "ok": not breaks,
        "n_checked": n_checked,
        "n_breaks": len(breaks),
        "breaks": breaks,
        "note": "first break in a mode identifies the tampered row; later breaks in the "
                "same mode cascade when a stored receipt itself was altered",
    }
