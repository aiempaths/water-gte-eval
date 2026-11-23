"""Baseline evaluation driver: one YAML config -> runs/<config_hash>/.

Loads prepared windows (parquet from `prepare`), fits each registered baseline on
the train stage, picks a flag threshold on the val stage (best F1), scores the
test stage, and reports shared metrics (window-level + event-level + bootstrap
CIs over events + ROC-AUC) per baseline. Deterministic: fixed seeds, sorted
outputs, no wall-clock fields.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score

from gte_eval.baselines import REGISTRY, make_baseline

from .config import config_hash
from .data import Window
from .metrics import bootstrap_event_metrics, build_event_records, event_metrics, window_metrics

META_COLS = {"window_id", "start", "end", "stage", "label", "event_id", "tag_group", "source_file"}


class BaselineEvalConfigError(Exception):
    pass


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise BaselineEvalConfigError(msg)


def load_baseline_config(path: str | Path) -> dict:
    p = Path(path)
    if not p.is_file():
        raise BaselineEvalConfigError(f"baseline config not found: {p}")
    try:
        raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise BaselineEvalConfigError(f"{p} is not valid YAML: {exc}") from exc
    _check(isinstance(raw, dict), "config root must be a mapping")
    _check(isinstance(raw.get("name"), str) and raw["name"], "'name' must be a non-empty string")
    _check(isinstance(raw.get("seed"), int), "'seed' must be an int")
    _check(isinstance(raw.get("windows_path"), str) and raw["windows_path"],
           "windows_path must point at a prepare *_windows.parquet")
    lb = raw.get("labels", {})
    _check(isinstance(lb.get("positive", "attack"), str), "labels.positive must be a string")
    bl = raw.get("baselines")
    _check(isinstance(bl, list) and bl and all(b in REGISTRY for b in bl),
           f"baselines must be a non-empty list drawn from {sorted(REGISTRY)}")
    _check(isinstance(raw.get("params", {}), dict), "params must be a mapping")
    th = raw.get("threshold", {})
    _check(th.get("mode", "val_best_f1") in ("val_best_f1", "conformal"),
           "threshold.mode must be 'val_best_f1' or 'conformal'")
    out = raw.get("output")
    _check(isinstance(out, dict) and isinstance(out.get("runs_dir"), str), "output.runs_dir required")
    raw["_path"] = str(p)
    raw["_config_hash"] = config_hash({k: v for k, v in raw.items() if not k.startswith("_")})
    return raw


def load_windows_parquet(path: str | Path) -> list[Window]:
    p = Path(path)
    if not p.is_file():
        raise BaselineEvalConfigError(
            f"windows file not found: {p}. Run `prepare` first (its *_windows.parquet) "
            "or fix windows_path."
        )
    df = pd.read_parquet(p)
    for c in ("window_id", "start", "end", "stage", "label"):
        if c not in df.columns:
            raise BaselineEvalConfigError(f"{p}: missing column {c!r}; expected a prepare windows parquet")
    tag_cols = [
        c for c in df.columns
        if c not in META_COLS and df[c].map(lambda v: isinstance(v, (list, np.ndarray))).all()
    ]
    if not tag_cols:
        raise BaselineEvalConfigError(f"{p}: no per-tag array columns found; columns: {list(df.columns)}")
    windows: list[Window] = []
    for _, row in df.iterrows():
        matrix = np.column_stack([np.asarray(row[t], dtype=np.float64) for t in tag_cols])
        windows.append(
            Window(
                window_id=str(row["window_id"]),
                start=pd.Timestamp(row["start"]),
                end=pd.Timestamp(row["end"]),
                matrix=matrix,
                sensors=tuple(tag_cols),
                label=str(row["label"]),
                scenario=str(row.get("event_id", "") or "background"),
                event_key=str(row["event_id"]) if str(row.get("event_id", "")) else None,
                event_id=str(row["event_id"]) if str(row.get("event_id", "")) else None,
                tag_group=str(row.get("tag_group", "default")),
                stage=str(row["stage"]),
            )
        )
    return sorted(windows, key=lambda w: (w.start, w.window_id))


def best_threshold(scores: list[float], y_binary: list[int]) -> tuple[float, float]:
    """Threshold (on score >= t) maximizing F1; ties -> smaller threshold."""
    candidates = sorted(set(float(s) for s in scores)) + [float("inf")]
    best_t, best_f1 = float("inf"), -1.0
    for t in candidates:
        tp = fp = fn = 0
        for s, y in zip(scores, y_binary):
            flag = s >= t
            if flag and y:
                tp += 1
            elif flag:
                fp += 1
            elif y:
                fn += 1
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
        if f1 > best_f1 or (f1 == best_f1 and t < best_t):
            best_t, best_f1 = t, f1
    return best_t, best_f1


def _dump(obj: dict) -> str:
    return json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n"


def _resolve(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else Path.cwd() / p


def run_baseline_eval(config_path: str | Path) -> Path:
    cfg = load_baseline_config(config_path)
    raw = {k: v for k, v in cfg.items() if not k.startswith("_")}
    run_hash = cfg["_config_hash"]
    out_dir = Path(cfg["output"]["runs_dir"]) / run_hash
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    windows = load_windows_parquet(_resolve(cfg["windows_path"]))
    positive = cfg.get("labels", {}).get("positive", "attack")
    stages = cfg.get("stages", {})
    fit_stage = stages.get("fit", "train")
    thr_stage = stages.get("threshold", "val")
    eval_stage = stages.get("eval", "test")
    by_stage: dict[str, list[Window]] = {}
    for w in windows:
        by_stage.setdefault(w.stage, []).append(w)
    for s in (fit_stage, thr_stage, eval_stage):
        if s not in by_stage or not by_stage[s]:
            raise BaselineEvalConfigError(f"stage {s!r} has no windows in {cfg['windows_path']}")
    train, val, test = by_stage[fit_stage], by_stage[thr_stage], by_stage[eval_stage]
    train_adapter = [w.to_adapter() for w in train]
    train_labels = [w.label for w in train]

    em_cfg = cfg.get("event_metrics", {})
    min_det = int(em_cfg.get("min_detected_windows", 1))
    bsec = cfg.get("bootstrap", {"unit": "events", "n_resamples": 1000, "ci": 0.95})
    boot_seed = int(bsec.get("seed", cfg["seed"]) or cfg["seed"])
    eval_days = (max(w.end for w in test) - min(w.start for w in test)).total_seconds() / 86400.0

    results: dict = {}
    score_rows: list[dict] = []
    scores_by_name: dict[str, dict] = {}
    flags_by_name: dict[str, dict] = {}
    for name in cfg["baselines"]:
        b = make_baseline({"name": name, "seed": cfg["seed"],
                           "params": cfg.get("params", {}).get(name, {})})
        b.fit(train_adapter, train_labels)
        val_scores = b.score_windows([w.to_adapter() for w in val])
        test_scores = b.score_windows([w.to_adapter() for w in test])

        y_val = [1 if w.label == positive else 0 for w in val]
        y_test = [1 if w.label == positive else 0 for w in test]
        if cfg.get("threshold", {}).get("mode", "val_best_f1") == "conformal" and name == "conformal":
            threshold = b.conformal_threshold(alpha=float(cfg["threshold"].get("alpha", 0.1)))
            thr_f1 = None
        else:
            threshold, thr_f1 = best_threshold(val_scores, y_val)
        val_flags = [s >= threshold for s in val_scores]
        test_flags = [s >= threshold for s in test_scores]

        val_rows = [{"label": w.label, "flag": f, "scenario": w.scenario} for w, f in zip(val, val_flags)]
        test_rows = [{"label": w.label, "flag": f, "scenario": w.scenario} for w, f in zip(test, test_flags)]
        wm_val = window_metrics(val_rows, positive)
        wm_test = window_metrics(test_rows, positive)

        flags_test = {w.window_id: f for w, f in zip(test, test_flags)}
        records = build_event_records(test, flags_test, positive, min_det)
        em = event_metrics(records, positive, eval_days)
        if records:
            boot = bootstrap_event_metrics(
                records, unit=bsec.get("unit", "events"),
                n_resamples=int(bsec.get("n_resamples", 1000)),
                ci=float(bsec.get("ci", 0.95)), seed=boot_seed, positive=positive,
            )
        else:
            boot = {"unit": bsec.get("unit", "events"), "n_resamples": 0,
                    "note": "no labeled events in the eval stage"}
        auc = None
        if len(set(y_test)) > 1:
            auc = float(roc_auc_score(y_test, test_scores))

        results[name] = {
            "params": b.describe(),
            "threshold": threshold,
            "val": {"f1_at_threshold": thr_f1, "window": wm_val},
            "test": {"roc_auc": auc, "window": wm_test, "event": em, "bootstrap": boot},
        }
        scores_by_name[name] = {}
        flags_by_name[name] = {}
        for w, s in zip(val, val_scores):
            score_rows.append({"window_id": w.window_id, "stage": "val", "label": w.label,
                               "event_id": w.event_id or "", "baseline": name, "score": float(s),
                               "flag": bool(s >= threshold)})
            scores_by_name[name][w.window_id] = float(s)
            flags_by_name[name][w.window_id] = bool(s >= threshold)
        for w, s in zip(test, test_scores):
            score_rows.append({"window_id": w.window_id, "stage": "test", "label": w.label,
                               "event_id": w.event_id or "", "baseline": name, "score": float(s),
                               "flag": bool(s >= threshold)})
            scores_by_name[name][w.window_id] = float(s)
            flags_by_name[name][w.window_id] = bool(s >= threshold)
        t = results[name]["test"]
        print(
            f"{name:18s} thr={threshold:.4g}  valF1={thr_f1 if thr_f1 is not None else 'n/a'}  "
            f"AUC={auc if auc is not None else 'n/a'}  testF1={t['window']['f1']}  "
            f"eventF1={t['event']['f1']}"
        )

    scores_df = pd.DataFrame(score_rows).sort_values(["stage", "window_id", "baseline"]).reset_index(drop=True)
    # wide frame: one row per window, one score/flag column per baseline
    meta = scores_df[["window_id", "stage", "label", "event_id"]].drop_duplicates("window_id")
    wide = meta.set_index("window_id")
    for name in cfg["baselines"]:
        wide[f"score__{name}"] = pd.Series(scores_by_name[name])
        wide[f"flag__{name}"] = pd.Series(flags_by_name[name])
    wide = wide.reset_index().sort_values(["stage", "window_id"]).reset_index(drop=True)

    scores_df.to_parquet(out_dir / "baseline_scores.parquet", index=False)
    wide.to_parquet(out_dir / "baseline_scores_wide.parquet", index=False)
    shutil.copyfile(cfg["_path"], out_dir / "config.yaml")
    (out_dir / "baselines.json").write_text(
        _dump({"config_hash": run_hash, "name": cfg["name"], "seed": cfg["seed"],
               "positive_label": positive, "stages": {"fit": fit_stage, "threshold": thr_stage,
                                                      "eval": eval_stage},
               "n_windows": {"train": len(train), "val": len(val), "test": len(test)},
               "results": results}), encoding="utf-8")
    print(f"run dir: {out_dir}")
    return out_dir
