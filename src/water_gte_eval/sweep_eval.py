"""Sweep driver: ONE command runs the engine evaluation across sweep axes and
reports each point as a table with event-bootstrapped CIs.

    python -m water_gte_eval sweeps configs/sweeps/example.yaml

Config schema (see configs/sweeps/example.yaml):
    name, seed
    base:      a full `eval` run-config body (inline `prepare:` so window length
               can be swept; `windows_path` fixes window length and is rejected
               for the window_length axis)
    axes:      k: [..]              -> engine.params.k
               window_length: [..]  -> prepare.window.length (samples)
               history_size: [..]   -> history.max_size (train history cap)
    output_dir: reports/sweeps     -> artifacts land in <output_dir>/<sweep_hash>/
    runs_dir:   runs               -> variant run dirs go to runs/<config_hash>/

Each variant is a full run config (written to <out>/variants/*.yaml and executed
through the canonical run_engine_eval path, so caching/resume/hashing all apply).
Metrics are computed by report.summarize_run_results — the same code as `report`.
Outputs: sweep.md (tables), sweep.json, one PNG per axis. Deterministic for a
fixed seed (no wall-clock fields).
"""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml

from .config import config_hash
from .engine_run import run_engine_eval
from .report import summarize_run_results

AXES = ("k", "window_length", "history_size")


class SweepConfigError(Exception):
    pass


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise SweepConfigError(msg)


def load_sweep_config(path: str | Path) -> dict:
    p = Path(path)
    if not p.is_file():
        raise SweepConfigError(f"sweep config not found: {p}")
    try:
        raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise SweepConfigError(f"{p} is not valid YAML: {exc}") from exc
    _check(isinstance(raw, dict), "config root must be a mapping")
    _check(isinstance(raw.get("name"), str) and raw["name"], "'name' required")
    _check(isinstance(raw.get("seed"), int), "'seed' must be an int")
    _check(isinstance(raw.get("base"), dict), "'base' must be a full run-config mapping")
    axes = raw.get("axes")
    _check(isinstance(axes, dict) and axes, "'axes' must be a non-empty mapping")
    for axis, values in axes.items():
        _check(axis in AXES, f"unknown sweep axis {axis!r}; allowed: {AXES}")
        _check(isinstance(values, list) and values, f"axes.{axis} must be a non-empty list")
    _check(isinstance(raw.get("output_dir"), str), "output_dir required")
    return raw


def _resolve(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else Path.cwd() / p


def _dump(obj: dict) -> str:
    return json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n"


def _apply_axis(cfg: dict, axis: str, value) -> None:
    if axis == "k":
        cfg["engine"]["params"]["k"] = int(value)
    elif axis == "window_length":
        if "prepare" not in cfg:
            raise SweepConfigError(
                "the window_length axis needs the base run config to build windows inline "
                "(`prepare:` section); with `windows_path` the window length is fixed"
            )
        cfg["prepare"]["window"]["length"] = int(value)
    elif axis == "history_size":
        cfg["history"] = {"max_size": int(value)}


def _row(axis, value, method: str, m: dict, budgets: list[float]) -> dict:
    boot = m["bootstrap"]

    def ci(key):
        b = boot.get(key) or {}
        return {"mean": b.get("mean"), "ci_low": b.get("ci_low"), "ci_high": b.get("ci_high")}

    return {
        "axis": axis, "value": value, "method": method,
        "auroc": m["auroc"], "auroc_ci": ci("auroc"),
        "auprc": m["auprc"], "auprc_ci": ci("auprc"),
        "event_fraction_detected": m["events"]["fraction_detected"],
        "event_fraction_detected_ci": ci("event_fraction_detected"),
        "false_trust_at_budget": {b: m["false_trust_at_budget"][b]["false_trust_rate"] for b in budgets},
        "false_trust_ci": ci(f"false_trust_rate@{budgets[0]}"),
        "escalations_per_hour": m["workload_escalations_per_hour"],
    }


def run_sweeps(config_path: str | Path) -> Path:
    cfg = load_sweep_config(config_path)
    base = cfg["base"]
    sweep_hash = config_hash({k: v for k, v in cfg.items() if k != "output_dir"})
    out_dir = _resolve(cfg["output_dir"]) / sweep_hash
    if out_dir.exists():
        shutil.rmtree(out_dir)
    variants_dir = out_dir / "variants"
    variants_dir.mkdir(parents=True)
    runs_dir = str(cfg.get("runs_dir", "runs"))
    positive_labels = list(cfg.get("positive_labels") or [base.get("positive_label", "attack")])
    boot = base.get("bootstrap", {})
    boot_n = int(boot.get("n_resamples", 200))
    ci = float(boot.get("ci", 0.95))
    seed = int(cfg["seed"])
    budgets = [0.05, 0.10, 0.20]

    rows: list[dict] = []
    runs: dict[str, str] = {}
    for axis, values in cfg["axes"].items():
        for value in values:
            suffix = f"{axis}={value}"
            variant = copy.deepcopy(base)
            variant["name"] = f"{base['name']}--{suffix}"
            _apply_axis(variant, axis, value)
            variant["output"] = {"runs_dir": runs_dir}
            vp = variants_dir / f"{suffix.replace('=', '__')}.yaml"
            vp.write_text(yaml.safe_dump(variant, sort_keys=True), encoding="utf-8")
            run_dir = run_engine_eval(vp)
            runs[suffix] = str(run_dir)
            summ = summarize_run_results(run_dir, positive_labels, budgets, boot_n, ci, seed)
            for method, m in summ["methods"].items():
                rows.append(_row(axis, value, method, m, budgets))
            print(f"sweep {suffix}: " + "  ".join(
                f"{meth} AUROC={m['auroc']:.3f}" if m["auroc"] is not None else f"{meth} AUROC=n/a"
                for meth, m in summ["methods"].items()))

    (out_dir / "sweep.json").write_text(_dump({
        "name": cfg["name"], "seed": seed, "sweep_hash": sweep_hash,
        "axes": cfg["axes"], "positive_labels": positive_labels,
        "bootstrap": {"n": boot_n, "ci": ci, "unit": "events", "seed": seed},
        "runs": runs, "table": rows,
    }), encoding="utf-8")

    # one PNG per axis: AUROC vs value per method, with bootstrap CI error bars
    for axis, values in cfg["axes"].items():
        methods = sorted({r["method"] for r in rows if r["axis"] == axis})
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for method in methods:
            pts = [r for r in rows if r["axis"] == axis and r["method"] == method]
            xs = [float(r["value"]) for r in pts]
            ys = [r["auroc"] if r["auroc"] is not None else float("nan") for r in pts]
            lo = [(y - (r["auroc_ci"]["ci_low"] or 0)) if y == y and r["auroc_ci"]["ci_low"] is not None else 0
                  for y, r in zip(ys, pts)]
            hi = [((r["auroc_ci"]["ci_high"] or 0) - y) if y == y and r["auroc_ci"]["ci_high"] is not None else 0
                  for y, r in zip(ys, pts)]
            ax.errorbar(xs, ys, yerr=[lo, hi], marker="o", capsize=3, label=method)
        ax.set_xlabel(axis)
        ax.set_ylabel("AUROC (test, event-bootstrap CI)")
        ax.set_title(f"sweep over {axis} \u2014 {cfg['name']}")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / f"sweep_{axis}.png", dpi=110)
        plt.close(fig)

    # sweep.md: one table per axis
    def _fmtc(x, nd=3):
        return "n/a" if x is None else f"{x:.{nd}f}"

    def _cic(b):
        if not b or b.get("mean") is None:
            return "n/a"
        return f"{b['mean']:.3f} [{b['ci_low']:.3f}, {b['ci_high']:.3f}]"

    lines = [f"# Sweep \u2014 {cfg['name']} (`{sweep_hash}`)", "",
             f"seed {seed}; bootstrap {boot_n} resamples over events, CI {ci}; "
             f"positive labels {positive_labels}; metrics on the **test** split only, "
             "scores rank-normalized on val (same code as `report`).", "",
             "Variant run configs: `variants/*.yaml`; run dirs: see `sweep.json -> runs`.", ""]
    for axis, values in cfg["axes"].items():
        lines.append(f"## axis `{axis}` = {list(values)}")
        lines.append("")
        lines.append("| value | method | AUROC (CI) | AUPRC (CI) | events detected (CI) | "
                     "false-trust @5/10/20% | escalations/h |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for r in [r for r in rows if r["axis"] == axis]:
            ftr = "/".join(_fmtc(r["false_trust_at_budget"][b]) for b in budgets)
            lines.append(f"| {r['value']} | {r['method']} | {_fmtc(r['auroc'])} ({_cic(r['auroc_ci'])}) "
                         f"| {_fmtc(r['auprc'])} ({_cic(r['auprc_ci'])}) "
                         f"| {_fmtc(r['event_fraction_detected'])} ({_cic(r['event_fraction_detected_ci'])}) "
                         f"| {ftr} ({_cic(r['false_trust_ci'])}) | {_fmtc(r['escalations_per_hour'], 2)} |")
        lines.append("")
    (out_dir / "sweep.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"sweep report: {out_dir / 'sweep.md'}  ({len(rows)} rows, {len(runs)} variant runs)")
    return out_dir
