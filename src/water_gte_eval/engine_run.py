"""Engine evaluation runner: one YAML config -> runs/<config_hash>/.

Pipeline: load/create windows + splits (from a prepare windows parquet, or
inline from a processed parquet via the `prepare:` section) -> build history
from TRAIN only, labeled per config `outcome` (next_window_state |
attack_normal | fault_type) -> score val/test through the EngineAdapter in the
configured modes (frozen | rolling | cold_start) -> run all configured baselines
on the SAME windows -> write results.parquet (one row per mode x window),
engine_log.parquet (per-window latency — kept out of results.parquet so re-runs
are byte-identical), metrics.json, manifest.json, config.yaml, environment.json.

Rules honored: history is train-only (plus, in rolling mode, previously SCORED
windows added only after scoring, with only labels knowable by then — never a
future/test label); thresholds are chosen on val, frozen, then applied to test;
resume + caching via per-mode cache files keyed by config hash + git commit +
windows-file hash.
"""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from gte_eval.engine_adapter.base import canonical_json

from .baseline_eval import best_threshold, load_windows_parquet
from .config import config_hash
from .data import Window
from .engine import AdapterWindow, HistoryWindow, make_engine
from .metrics import bootstrap_event_metrics, build_event_records, event_metrics, window_metrics
from gte_eval.baselines import REGISTRY, make_baseline

MODES = ("frozen", "rolling", "cold_start")
OUTCOMES = ("next_window_state", "attack_normal", "fault_type")


class RunConfigError(Exception):
    pass


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise RunConfigError(msg)


def load_run_config(path: str | Path) -> dict:
    p = Path(path)
    if not p.is_file():
        raise RunConfigError(f"run config not found: {p}")
    try:
        raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise RunConfigError(f"{p} is not valid YAML: {exc}") from exc
    _check(isinstance(raw, dict), "config root must be a mapping")
    _check(isinstance(raw.get("name"), str) and raw["name"], "'name' required")
    _check(isinstance(raw.get("seed"), int), "'seed' must be an int")
    _check(("windows_path" in raw) or ("prepare" in raw), "windows_path or a prepare: section is required")
    _check(isinstance(raw.get("engine"), dict) and raw["engine"].get("type"), "engine.type required")
    modes = raw.get("modes", ["frozen"])
    _check(isinstance(modes, list) and modes and all(m in MODES for m in modes),
           f"modes must be a non-empty subset of {MODES}")
    _check(raw.get("outcome", "attack_normal") in OUTCOMES,
           f"outcome must be one of {OUTCOMES}")
    _check(isinstance(raw.get("baselines", []), list)
           and all(b in REGISTRY for b in raw.get("baselines", [])),
           f"baselines must be drawn from {sorted(REGISTRY)}")
    out = raw.get("output")
    _check(isinstance(out, dict) and isinstance(out.get("runs_dir"), str), "output.runs_dir required")
    raw["_path"] = str(p)
    raw["_config_hash"] = config_hash({k: v for k, v in raw.items() if not k.startswith("_")})
    return raw


def _resolve(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else Path.cwd() / p


def _dump(obj: dict) -> str:
    return json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n"


def git_commit() -> str:
    try:
        root = Path(__file__).resolve().parents[2]
        out = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return "unknown"


def environment_info(commit: str) -> dict:
    import pyarrow
    import sklearn
    import wntr

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "git_commit": commit,
        "packages": {
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit-learn": sklearn.__version__,
            "pyarrow": pyarrow.__version__,
            "wntr": wntr.__version__,
        },
    }


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- windows

def _load_or_build_windows(cfg: dict) -> tuple[list[Window], str]:
    """Returns (windows with stage/label/event_id set, windows source sha256 or '')."""
    if cfg.get("windows_path"):
        p = _resolve(cfg["windows_path"])
        windows = load_windows_parquet(p)
        return windows, _file_sha256(p)
    prep = cfg["prepare"]
    from .prepare import _load_intervals, _prepare_one

    src = prep["source"]
    src_path = _resolve(src["path"])
    if src_path.is_file():
        files = [src_path]
    elif src_path.is_dir():
        files = sorted(src_path.glob("*.parquet"))
    else:
        raise RunConfigError(f"prepare.source.path not found: {src_path}")
    if not files:
        raise RunConfigError(f"no parquet in {src_path}")
    intervals = _load_intervals(prep)
    all_windows: list[Window] = []
    for f in files:
        df = pd.read_parquet(f)
        res = _prepare_one(df, prep, intervals, f.name)
        all_windows.extend(res["windows"])
    all_windows.sort(key=lambda w: (w.start, w.window_id))
    return all_windows, _file_sha256(files[0]) if len(files) == 1 else "multi"


# ---------------------------------------------------------------- outcome

def _outcome_fn(outcome: str, positive: str):
    """Map a window to the history outcome label. next_window_state uses the NEXT
    window (by window_id order) in the SAME STAGE list only; the last one falls
    back to its own label. Crossing into the next stage is deliberately NOT done:
    that would let val/test labels leak into train history. Keyed by window_id
    (Window contains ndarrays, not eq-comparable)."""
    def fn(w: Window, windows_list: list[Window]) -> str:
        if outcome == "fault_type":
            return w.label
        if outcome == "attack_normal":
            return "attack" if w.label == positive else "normal"
        ids = [x.window_id for x in windows_list]
        i = ids.index(w.window_id)
        if i + 1 < len(windows_list):
            return windows_list[i + 1].label
        return w.label
    return fn


def _history_window(w: Window, label: str) -> HistoryWindow:
    aw = w.to_adapter()
    return HistoryWindow(window_id=aw.window_id, timestamps=list(aw.timestamps),
                         sensors={k: list(v) for k, v in aw.sensors.items()},
                         outcome_label=label)


# ---------------------------------------------------------------- engine modes

def _run_mode(mode: str, engine_cfg: dict, train: list[Window], scored: list[Window],
              outcome: str, positive: str, seed: int, cold_n: int) -> dict:
    """Score all scored windows (val+test, time order) in one history mode.

    History rule: a window enters the engine history only when its end is
    STRICTLY before the scored window's start, and (rolling mode) only after it
    has been scored, with an outcome label that is knowable by then
    (next_window_state waits until the next window has happened).
    Returns {window_id: {closeness, agreement, verdict, receipt, neighbor_ids,
    latency_ms}}."""
    engine = make_engine(engine_cfg)
    engine.reset_history()
    outcome_fn = _outcome_fn(outcome, positive)

    hist_pool = sorted(train, key=lambda w: (w.start, w.window_id))
    if mode == "cold_start" and cold_n and len(hist_pool) > cold_n:
        rng = np.random.default_rng(seed)
        keep = sorted(rng.choice(len(hist_pool), size=cold_n, replace=False).tolist())
        hist_pool = [hist_pool[i] for i in keep]

    scored_sorted = sorted(scored, key=lambda w: (w.start, w.window_id))
    scored_ids = [w.window_id for w in scored_sorted]
    next_of = {scored_ids[i]: (scored_sorted[i + 1] if i + 1 < len(scored_sorted) else None)
               for i in range(len(scored_ids))}
    train_ids = {w.window_id for w in train}

    pool = list(hist_pool)          # candidate history windows (train, or scored in rolling)
    added: set[str] = set()
    results: dict[str, dict] = {}
    for w in scored_sorted:
        # add every pool window that is eligible now (end < w.start) and whose
        # outcome label is knowable now
        for h in pool:
            if h.window_id in added or h.end >= w.start:
                continue
            if h.window_id not in train_ids and outcome == "next_window_state":
                nxt = next_of[h.window_id]
                if nxt is not None and nxt.start >= w.start:
                    continue  # next window hasn't happened yet -> label not knowable
                label = nxt.label if nxt is not None else h.label
            else:
                label = outcome_fn(h, train if h.window_id in train_ids else scored_sorted)
            engine.add_history([_history_window(h, label)])
            added.add(h.window_id)
        res = engine.score(w.to_adapter())
        results[w.window_id] = {
            "closeness": res.closeness,
            "agreement": res.agreement,
            "verdict": res.verdict,
            "receipt": res.receipt,
            "neighbor_ids": list(res.neighbor_ids),
            "latency_ms": res.latency_ms,
        }
        if mode == "rolling":
            pool.append(w)  # only AFTER scoring
    return results


# ---------------------------------------------------------------- cache

class _ModeCache:
    def __init__(self, out_dir: Path, resume: bool):
        self.dir = out_dir / ".cache"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.resume = resume
        self.hits: list[str] = []
        self.misses: list[str] = []

    def _valid(self, path: Path, git: str, windows_sha: str) -> bool:
        if not (self.resume and path.is_file()):
            return False
        try:
            meta = json.loads(path.read_text())
            return meta.get("git_commit") == git and meta.get("windows_sha256") == windows_sha
        except Exception:
            return False

    def load(self, key: str, git: str, windows_sha: str):
        path = self.dir / f"{key}.json"
        if self._valid(path, git, windows_sha):
            self.hits.append(key)
            return json.loads(path.read_text())["payload"]
        self.misses.append(key)
        return None

    def store(self, key: str, git: str, windows_sha: str, payload) -> None:
        path = self.dir / f"{key}.json"
        path.write_text(_dump({"git_commit": git, "windows_sha256": windows_sha,
                               "payload": payload}))


# ---------------------------------------------------------------- main

def run_engine_eval(config_path: str | Path) -> Path:
    cfg = load_run_config(config_path)
    raw = {k: v for k, v in cfg.items() if not k.startswith("_")}
    run_hash = cfg["_config_hash"]
    out_dir = Path(cfg["output"]["runs_dir"]) / run_hash
    resume = bool(cfg.get("resume", True))
    if out_dir.exists() and not resume:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    commit = git_commit()
    windows, windows_sha = _load_or_build_windows(cfg)
    by_stage: dict[str, list[Window]] = {}
    for w in windows:
        by_stage.setdefault(w.stage, []).append(w)
    for s in ("train", "val", "test"):
        if s not in by_stage or not by_stage[s]:
            raise RunConfigError(f"stage {s!r} missing/empty; windows must come from prepare with stages")
    train, val, test = by_stage["train"], by_stage["val"], by_stage["test"]
    scored = sorted(val + test, key=lambda w: (w.start, w.window_id))
    positive = cfg.get("positive_label", "attack")
    outcome = cfg.get("outcome", "attack_normal")
    seed = int(cfg["seed"])
    cold_n = int(cfg.get("cold_start", {}).get("n_history", 0))

    cache = _ModeCache(out_dir, resume)

    # engine modes
    engine_results: dict[str, dict[str, dict]] = {}
    for mode in cfg["modes"]:
        key = f"engine_{mode}"
        cached = cache.load(key, commit, windows_sha)
        if cached is not None:
            engine_results[mode] = cached
        else:
            engine_results[mode] = _run_mode(mode, cfg["engine"], train, scored, outcome,
                                             positive, seed, cold_n)
            cache.store(key, commit, windows_sha, engine_results[mode])

    # baselines on identical windows (fit on train only)
    baseline_names = cfg.get("baselines", [])
    baseline_scores: dict[str, dict[str, float]] = {}
    bkey = "baselines"
    cached = cache.load(bkey, commit, windows_sha)
    if cached is not None:
        baseline_scores = cached
    else:
        train_adapter = [w.to_adapter() for w in train]
        outcome_fn = _outcome_fn(outcome, positive)
        train_labels = [outcome_fn(w, train) for w in train]
        for name in baseline_names:
            b = make_baseline({"name": name, "seed": seed,
                               "params": cfg.get("baseline_params", {}).get(name, {})})
            b.fit(train_adapter, train_labels)
            baseline_scores[name] = {w.window_id: s for w, s in
                                     zip(scored, b.score_windows([w.to_adapter() for w in scored]))}
        cache.store(bkey, commit, windows_sha, baseline_scores)

    # thresholds on VAL only, frozen, then applied to test
    val_ids = [w.window_id for w in val]
    test_ids = [w.window_id for w in test]
    thresholds: dict[str, float] = {}
    for name in baseline_names:
        scores = baseline_scores[name]
        y_val = [1 if w.label == positive else 0 for w in val]
        thresholds[name], _ = best_threshold([scores[wid] for wid in val_ids], y_val)

    # results rows: one per (mode, scored window)
    rows = []
    for mode in cfg["modes"]:
        for w in scored:
            r = engine_results[mode][w.window_id]
            flag = r["verdict"] in ("route_to_human", "unavailable")
            row = {
                "config_hash": run_hash,
                "git_commit": commit,
                "mode": mode,
                "split": w.stage,
                "window_id": w.window_id,
                "start": w.start,
                "end": w.end,
                "event_id": w.event_id or "",
                "true_label": w.label,
                "engine_closeness": r["closeness"],
                "engine_agreement": r["agreement"],
                "engine_verdict": r["verdict"],
                "engine_receipt": r["receipt"],
                "engine_neighbors": r["neighbor_ids"],
                "engine_flag": bool(flag),
            }
            for name in baseline_names:
                s = baseline_scores[name][w.window_id]
                row[f"baseline__{name}"] = float(s)
                row[f"baseline__{name}__flag"] = bool(s >= thresholds[name])
            rows.append(row)
    results_df = pd.DataFrame(rows).sort_values(["mode", "split", "window_id"]).reset_index(drop=True)
    results_df.to_parquet(out_dir / "results.parquet", index=False)

    # engine log with latency (kept separate: wall-clock, not byte-deterministic)
    log_rows = []
    for mode in cfg["modes"]:
        for w in scored:
            r = engine_results[mode][w.window_id]
            log_rows.append({"mode": mode, "split": w.stage, "window_id": w.window_id,
                             "latency_ms": r["latency_ms"], "verdict": r["verdict"],
                             "receipt": r["receipt"]})
    pd.DataFrame(log_rows).sort_values(["mode", "split", "window_id"]).to_parquet(
        out_dir / "engine_log.parquet", index=False)

    # metrics per mode (engine) + per baseline (shared metrics code)
    em_cfg = cfg.get("event_metrics", {})
    min_det = int(em_cfg.get("min_detected_windows", 1))
    bsec = cfg.get("bootstrap", {"unit": "events", "n_resamples": 1000, "ci": 0.95})
    boot_seed = int(bsec.get("seed", seed) or seed)
    eval_days = (max(w.end for w in test) - min(w.start for w in test)).total_seconds() / 86400.0

    def _flag_dict(mode):
        return {w.window_id: engine_results[mode][w.window_id]["verdict"] in ("route_to_human", "unavailable")
                for w in scored}

    metrics: dict = {"config_hash": run_hash, "git_commit": commit, "positive_label": positive,
                     "outcome": outcome, "threshold_policy": "chosen on val, frozen, applied to test",
                     "modes": {}, "baselines": {}}
    for mode in cfg["modes"]:
        flags = _flag_dict(mode)
        mode_metrics = {}
        for stage, ws in (("val", val), ("test", test)):
            rows_s = [{"label": w.label, "flag": flags[w.window_id], "scenario": w.scenario} for w in ws]
            wm = window_metrics(rows_s, positive)
            recs = build_event_records(ws, flags, positive, min_det)
            em = event_metrics(recs, positive, eval_days)
            boot = bootstrap_event_metrics(recs, unit=bsec.get("unit", "events"),
                                           n_resamples=int(bsec.get("n_resamples", 1000)),
                                           ci=float(bsec.get("ci", 0.95)), seed=boot_seed,
                                           positive=positive) if recs else {"note": "no events"}
            mode_metrics[stage] = {"window": wm, "event": em, "bootstrap": boot}
        metrics["modes"][mode] = mode_metrics
    for name in baseline_names:
        t = thresholds[name]
        bm = {}
        for stage, ws in (("val", val), ("test", test)):
            rows_s = [{"label": w.label,
                       "flag": baseline_scores[name][w.window_id] >= t,
                       "scenario": w.scenario} for w in ws]
            wm = window_metrics(rows_s, positive)
            recs = build_event_records(ws, {w.window_id: baseline_scores[name][w.window_id] >= t
                                            for w in ws}, positive, min_det)
            em = event_metrics(recs, positive, eval_days)
            boot = bootstrap_event_metrics(recs, unit=bsec.get("unit", "events"),
                                           n_resamples=int(bsec.get("n_resamples", 1000)),
                                           ci=float(bsec.get("ci", 0.95)), seed=boot_seed,
                                           positive=positive) if recs else {"note": "no events"}
            bm[stage] = {"window": wm, "event": em, "bootstrap": boot}
        metrics["baselines"][name] = {"threshold": t, "threshold_source": "val", **bm}
        t_row = metrics["baselines"][name]["test"]["window"]
        print(f"baseline {name:18s} thr={t:.4g} testF1={t_row['f1']}")
    for mode in cfg["modes"]:
        t_row = metrics["modes"][mode]["test"]["window"]
        e_row = metrics["modes"][mode]["test"]["event"]
        print(f"engine mode={mode:10s} testF1={t_row['f1']} eventF1={e_row['f1']}")

    (out_dir / "metrics.json").write_text(_dump(metrics), encoding="utf-8")
    shutil.copyfile(cfg["_path"], out_dir / "config.yaml")
    (out_dir / "environment.json").write_text(_dump(environment_info(commit)), encoding="utf-8")

    outputs = ["config.yaml", "results.parquet", "engine_log.parquet", "metrics.json",
               "environment.json"]
    manifest = {
        "config_hash": run_hash,
        "name": cfg["name"],
        "seed": seed,
        "git_commit": commit,
        "engine": cfg["engine"],
        "modes": cfg["modes"],
        "outcome": outcome,
        "resume": resume,
        "cache": {"hits": cache.hits, "computed": cache.misses},
        "n_windows": {"train": len(train), "val": len(val), "test": len(test)},
        "thresholds": thresholds,
        "files": {name: _file_sha256(out_dir / name) for name in outputs},
    }
    (out_dir / "manifest.json").write_text(_dump(manifest), encoding="utf-8")
    print(f"run dir: {out_dir}")
    return out_dir
