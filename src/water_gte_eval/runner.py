"""Experiment runner: one YAML config in, one deterministic run directory out.

Pipeline: load config -> hash it -> load and validate real data files -> build
time windows -> temporal/scenario split -> score eval windows through the
EngineAdapter (history strictly before each scored window) -> decide
trust/route-to-human -> per-window and event-level metrics -> bootstrap CIs
over events/scenarios -> write runs/<config_hash>/.

Outputs contain no wall-clock timestamps, so re-running the same config
reproduces byte-identical files.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

from .config import Config, config_hash, load_config
from .data import (
    EventColumns,
    TelemetryColumns,
    build_windows,
    eligible_history,
    file_sha256,
    load_table,
    require_columns,
    temporal_split,
)
from .engine import EngineError, make_engine
from .engine.base import ScoreResult
from .metrics import bootstrap_event_metrics, build_event_records, event_metrics, per_scenario, window_metrics


def _resolve(p: str | Path) -> Path:
    """Relative paths resolve against the current working directory; run
    experiments from the repo root."""
    p = Path(p)
    return p if p.is_absolute() else Path.cwd() / p


def _decide(result: ScoreResult, decision_cfg: dict) -> bool:
    """Downstream decision logic: close+agreeing = trust; far or mixed =
    route to human (flag)."""
    if decision_cfg["mode"] == "verdict":
        return result.verdict == decision_cfg.get("route_verdict", "route_to_human")
    if result.closeness is None or result.agreement is None:
        return True
    return bool(
        result.closeness > float(decision_cfg["max_closeness"])
        or result.agreement < float(decision_cfg["min_agreement"])
    )


def _dump(obj: dict) -> str:
    return json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n"


def run_experiment(config_path: str | Path) -> Path:
    cfg: Config = load_config(config_path)
    raw = cfg.raw
    run_hash = config_hash(raw)
    out_dir = Path(raw["output"]["runs_dir"]) / run_hash
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    dsec, esec = raw["dataset"], raw["events"]
    dpath, epath = _resolve(dsec["path"]), _resolve(esec["path"])
    tdf = load_table(dpath, dsec.get("format"))
    edf = load_table(epath, esec.get("format"))
    tcols = TelemetryColumns.from_config(dsec["columns"])
    ecols = EventColumns.from_config(esec["columns"])
    require_columns(tdf, [tcols.timestamp, *tcols.sensors], f"Dataset file {dpath}")
    require_columns(edf, [ecols.scenario, ecols.start, ecols.end, ecols.label],
                    f"Events file {epath}")
    tdf = tdf.copy()
    tdf[tcols.timestamp] = pd.to_datetime(tdf[tcols.timestamp])

    wsec = raw["window"]
    size, step = pd.Timedelta(wsec["size"]), pd.Timedelta(wsec["step"])
    sp = raw["split"]
    history_end = pd.Timestamp(sp["history_end"])
    eval_start = pd.Timestamp(sp["eval_start"])
    eval_end = pd.Timestamp(sp["eval_end"])
    labels = raw["labels"]
    positive, default_label = labels["positive"], labels["default"]

    data_start = pd.Timestamp(tdf[tcols.timestamp].min())
    windows = build_windows(
        tdf, edf, tcols, ecols, size, step, data_start, eval_end, default_label
    )
    split = temporal_split(
        windows, history_end, eval_start, eval_end, sp.get("holdout_scenarios", [])
    )

    engine = make_engine(raw["engine"])
    decision_cfg = raw["decision"]
    em_cfg = raw["event_metrics"]
    min_detected = int(em_cfg.get("min_detected_windows", 1))

    pool = split.history
    added: set[str] = set()
    rows: list[dict] = []
    for w in split.eval:
        for hw in eligible_history(pool, w):
            if hw.window_id not in added:
                engine.add_to_history(hw, hw.label)
                added.add(hw.window_id)
        result = engine.score(w)
        if not engine.verify_receipt(w, result):
            raise EngineError(f"receipt verification failed for window {w.window_id}")
        flag = _decide(result, decision_cfg)
        rows.append(
            {
                "window_id": w.window_id,
                "start": w.start,
                "end": w.end,
                "scenario": w.scenario,
                "label": w.label,
                "event_key": w.event_key if w.event_key is not None else "",
                "flag": bool(flag),
                "closeness": result.closeness,
                "agreement": result.agreement,
                "verdict": result.verdict,
                "receipt": result.receipt,
            }
        )
    preds = pd.DataFrame(rows)

    flags = {r["window_id"]: r["flag"] for r in rows}
    records = build_event_records(split.eval, flags, positive, min_detected)
    eval_days = (eval_end - eval_start).total_seconds() / 86400.0
    wm = window_metrics(rows, positive)
    wm["per_scenario"] = per_scenario(rows, positive)
    em = event_metrics(records, positive, eval_days)

    bsec = raw["bootstrap"]
    boot = bootstrap_event_metrics(
        records,
        unit=bsec["unit"],
        n_resamples=int(bsec["n_resamples"]),
        ci=float(bsec["ci"]),
        seed=int(bsec["seed"]) if bsec.get("seed") is not None else cfg.seed,
        positive=positive,
    )
    metrics = {
        "positive_label": positive,
        "decision": decision_cfg,
        "window": wm,
        "event": em,
        "bootstrap": boot,
    }

    events_df = pd.DataFrame(
        [
            {
                "event_key": r.event_key,
                "scenario": r.scenario,
                "label": r.label,
                "n_windows": r.n_windows,
                "n_flagged": r.n_flagged,
                "detected": r.detected,
            }
            for r in records
        ]
    )

    shutil.copyfile(cfg.path, out_dir / "config.yaml")
    preds.to_parquet(out_dir / "predictions.parquet", index=False)
    preds.to_csv(out_dir / "predictions.csv", index=False)
    events_df.to_csv(out_dir / "events.csv", index=False)
    (out_dir / "metrics.json").write_text(_dump(metrics), encoding="utf-8")

    manifest = {
        "config_hash": run_hash,
        "config_name": cfg.name,
        "config_file": str(cfg.path),
        "seed": cfg.seed,
        "engine": {
            **engine.describe(),
            "receipt_chain_head": engine.receipt_chain_head,
            "receipts_verified": True,
        },
        "dataset": {
            "path": str(dpath),
            "sha256": file_sha256(dpath),
            "n_rows": int(len(tdf)),
            "timestamp_column": tcols.timestamp,
            "sensor_columns": list(tcols.sensors),
            "time_min": pd.Timestamp(tdf[tcols.timestamp].min()).isoformat(),
            "time_max": pd.Timestamp(tdf[tcols.timestamp].max()).isoformat(),
        },
        "events": {
            "path": str(epath),
            "sha256": file_sha256(epath),
            "n_rows": int(len(edf)),
            "columns": {
                "scenario": ecols.scenario,
                "start": ecols.start,
                "end": ecols.end,
                "label": ecols.label,
            },
        },
        "window": {"size": wsec["size"], "step": wsec["step"]},
        "split": {
            "history_end": history_end.isoformat(),
            "eval_start": eval_start.isoformat(),
            "eval_end": eval_end.isoformat(),
            "holdout_scenarios": list(sp.get("holdout_scenarios", [])),
            "n_history_windows": len(split.history),
            "n_eval_windows": len(split.eval),
        },
        "outputs": [
            "config.yaml",
            "manifest.json",
            "predictions.parquet",
            "predictions.csv",
            "events.csv",
            "metrics.json",
        ],
    }
    files = {name: file_sha256(out_dir / name) for name in manifest["outputs"] if name != "manifest.json"}
    manifest["files"] = files
    (out_dir / "manifest.json").write_text(_dump(manifest), encoding="utf-8")
    return out_dir
