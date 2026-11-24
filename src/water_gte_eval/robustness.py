"""Robustness driver: run the engine eval under telemetry perturbations and a
poisoned history, and report how scores/metrics degrade vs the clean run.

    python -m water_gte_eval robustness configs/robustness/example.yaml

Config schema (see configs/robustness/example.yaml):
    name, seed
    base:            a full `eval` run-config body
    perturbations:   list of perturb sections, e.g.
                     - {name: sensor_dropout, params: {prob: 0.3, fill: hold_first}, scope: scored}
                     - {name: clock_skew, params: {shift_samples: 2}, scope: scored}
    poison_levels:   [0.0, 0.3, ...]  -> poison_history_frac on the ENGINE history
    output_dir:      reports/robustness
    runs_dir:        runs

Variants: one clean run, one per perturbation (poison 0), one per poison level
(no perturbation). Every variant is a full run config executed through the
canonical run_engine_eval path (perturb/poison are part of the config hash, so
variant run dirs never collide). Metrics come from report.summarize_run_results.
Outputs: robustness.md, robustness.json, degradation PNG. Deterministic.
"""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

from .config import config_hash
from .engine_run import run_engine_eval
from .perturb import PERTURBATIONS
from .report import summarize_run_results


class RobustnessConfigError(Exception):
    pass


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise RobustnessConfigError(msg)


def load_robustness_config(path: str | Path) -> dict:
    p = Path(path)
    if not p.is_file():
        raise RobustnessConfigError(f"robustness config not found: {p}")
    try:
        raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise RobustnessConfigError(f"{p} is not valid YAML: {exc}") from exc
    _check(isinstance(raw, dict), "config root must be a mapping")
    _check(isinstance(raw.get("name"), str) and raw["name"], "'name' required")
    _check(isinstance(raw.get("seed"), int), "'seed' must be an int")
    _check(isinstance(raw.get("base"), dict), "'base' must be a full run-config mapping")
    perts = raw.get("perturbations", [])
    _check(isinstance(perts, list), "'perturbations' must be a list")
    for pert in perts:
        _check(isinstance(pert, dict) and pert.get("name") in PERTURBATIONS,
               f"each perturbation needs name in {PERTURBATIONS}")
    poisons = raw.get("poison_levels", [0.0])
    _check(isinstance(poisons, list) and all(isinstance(x, (int, float)) and 0.0 <= float(x) <= 1.0
                                             for x in poisons), "poison_levels must be floats in [0,1]")
    _check(isinstance(raw.get("output_dir"), str), "output_dir required")
    return raw


def _resolve(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else Path.cwd() / p


def _dump(obj: dict) -> str:
    return json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n"


def run_robustness(config_path: str | Path) -> Path:
    cfg = load_robustness_config(config_path)
    base = cfg["base"]
    seed = int(cfg["seed"])
    rob_hash = config_hash({k: v for k, v in cfg.items() if k != "output_dir"})
    out_dir = _resolve(cfg["output_dir"]) / rob_hash
    if out_dir.exists():
        shutil.rmtree(out_dir)
    variants_dir = out_dir / "variants"
    variants_dir.mkdir(parents=True)
    runs_dir = str(cfg.get("runs_dir", "runs"))
    positive_labels = list(cfg.get("positive_labels") or [base.get("positive_label", "attack")])
    boot = base.get("bootstrap", {})
    boot_n = int(boot.get("n_resamples", 200))
    ci = float(boot.get("ci", 0.95))
    budgets = [0.05, 0.10, 0.20]

    variants: list[tuple[str, dict]] = [("clean", {})]
    for i, pert in enumerate(cfg.get("perturbations", [])):
        variants.append((f"perturb{i}:{pert['name']}", {"perturb": pert}))
    for p in cfg.get("poison_levels", [0.0]):
        if float(p) > 0.0:
            variants.append((f"poison={p}", {"poison_history_frac": float(p)}))

    rows: list[dict] = []
    runs: dict[str, str] = {}
    for name, over in variants:
        variant = copy.deepcopy(base)
        variant["name"] = f"{base['name']}--{name}"
        variant.update(over)
        variant["output"] = {"runs_dir": runs_dir}
        vp = variants_dir / f"{name.replace(':', '__').replace('=', '__').replace('/', '_')}.yaml"
        vp.write_text(yaml.safe_dump(variant, sort_keys=True), encoding="utf-8")
        run_dir = run_engine_eval(vp)
        runs[name] = str(run_dir)
        summ = summarize_run_results(run_dir, positive_labels, budgets, boot_n, ci, seed)
        for method, m in summ["methods"].items():
            b = m["bootstrap"]
            rows.append({
                "variant": name, "method": method,
                "auroc": m["auroc"],
                "auroc_ci": {"mean": b["auroc"].get("mean"), "ci_low": b["auroc"].get("ci_low"),
                             "ci_high": b["auroc"].get("ci_high")},
                "auprc": m["auprc"],
                "event_fraction_detected": m["events"]["fraction_detected"],
                "event_fraction_detected_ci": {"mean": b["event_fraction_detected"].get("mean"),
                                               "ci_low": b["event_fraction_detected"].get("ci_low"),
                                               "ci_high": b["event_fraction_detected"].get("ci_high")},
                "false_trust_at_budget": {bb: m["false_trust_at_budget"][bb]["false_trust_rate"] for bb in budgets},
                "escalations_per_hour": m["workload_escalations_per_hour"],
            })
        eng_rows = {r["method"]: r for r in rows if r["variant"] == name and r["method"].startswith("engine__")}
        print(f"robustness {name}: " + "  ".join(
            f"{meth} AUROC={r['auroc']:.3f}" if r["auroc"] is not None else f"{meth} AUROC=n/a"
            for meth, r in eng_rows.items()))

    # delta vs clean (point estimate) per variant/method
    clean = {(r["method"]): r for r in rows if r["variant"] == "clean"}
    for r in rows:
        c = clean.get(r["method"])
        if c and c["auroc"] is not None and r["auroc"] is not None:
            r["delta_auroc_vs_clean"] = r["auroc"] - c["auroc"]
        else:
            r["delta_auroc_vs_clean"] = None

    (out_dir / "robustness.json").write_text(_dump({
        "name": cfg["name"], "seed": seed, "robustness_hash": rob_hash,
        "positive_labels": positive_labels,
        "bootstrap": {"n": boot_n, "ci": ci, "unit": "events", "seed": seed},
        "variants": [v[0] for v in variants], "runs": runs, "table": rows,
    }), encoding="utf-8")

    def _fmtc(x, nd=3):
        return "n/a" if x is None else f"{x:.{nd}f}"

    def _cic(b):
        if not b or b.get("mean") is None:
            return "n/a"
        return f"{b['mean']:.3f} [{b['ci_low']:.3f}, {b['ci_high']:.3f}]"

    lines = [f"# Robustness \u2014 {cfg['name']} (`{rob_hash}`)", "",
             f"seed {seed}; bootstrap {boot_n} resamples over events, CI {ci}; test split only; "
             "each variant is a full `eval` run (configs in `variants/`, run dirs in `robustness.json`).", "",
             "Perturbations apply to scored (val+test) windows only (`scope: scored`), so train "
             "history stays clean; `poison_levels` mislabel that fraction of TRAIN attack windows "
             "as `normal` in the ENGINE history only (baselines are not poisoned).", ""]
    engine_methods = sorted({r["method"] for r in rows if r["method"].startswith("engine__")})
    lines.append("## Engine metrics per variant (AUROC / event detection with CIs; \u0394 = vs clean)")
    lines.append("")
    lines.append("| variant | method | AUROC (CI) | \u0394AUROC | events detected (CI) | false-trust @5/10/20% |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for name, _ in variants:
        for method in engine_methods:
            r = next((x for x in rows if x["variant"] == name and x["method"] == method), None)
            if r is None:
                continue
            ftr = "/".join(_fmtc(r["false_trust_at_budget"][b]) for b in budgets)
            lines.append(f"| {name} | {method} | {_fmtc(r['auroc'])} ({_cic(r['auroc_ci'])}) "
                         f"| {_fmtc(r['delta_auroc_vs_clean'])} "
                         f"| {_fmtc(r['event_fraction_detected'])} ({_cic(r['event_fraction_detected_ci'])}) "
                         f"| {ftr} |")
    lines.append("")
    lines.append("## All methods (incl. baselines, unpoisoned) per variant")
    lines.append("")
    lines.append("| variant | method | AUROC | AUPRC | events detected | \u0394AUROC vs clean |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for name, _ in variants:
        for r in [x for x in rows if x["variant"] == name]:
            lines.append(f"| {name} | {r['method']} | {_fmtc(r['auroc'])} | {_fmtc(r['auprc'])} "
                         f"| {_fmtc(r['event_fraction_detected'])} | {_fmtc(r['delta_auroc_vs_clean'])} |")
    lines.append("")
    (out_dir / "robustness.md").write_text("\n".join(lines), encoding="utf-8")

    # degradation figure: AUROC per variant per engine method with CI
    fig, ax = plt.subplots(figsize=(max(7, 1.2 * len(variants)), 4.5))
    x = range(len(variants))
    w = 0.8 / max(1, len(engine_methods))
    for j, method in enumerate(engine_methods):
        ys, lo, hi = [], [], []
        for name, _ in variants:
            r = next((x for x in rows if x["variant"] == name and x["method"] == method), None)
            ys.append(r["auroc"] if r and r["auroc"] is not None else float("nan"))
            ci_low = r["auroc_ci"]["ci_low"] if r else None
            ci_high = r["auroc_ci"]["ci_high"] if r else None
            lo.append((ys[-1] - ci_low) if ci_low is not None and ys[-1] == ys[-1] else 0.0)
            hi.append((ci_high - ys[-1]) if ci_high is not None and ys[-1] == ys[-1] else 0.0)
        ax.bar([i + (j - (len(engine_methods) - 1) / 2) * w for i in x], ys,
               width=w, yerr=[lo, hi], capsize=2, label=method)
    ax.set_xticks(list(x))
    ax.set_xticklabels([v[0] for v in variants], rotation=30, ha="right")
    ax.set_ylabel("AUROC (test, event-bootstrap CI)")
    ax.set_title(f"robustness \u2014 {cfg['name']}")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / "robustness_degradation.png", dpi=110)
    plt.close(fig)
    print(f"robustness report: {out_dir / 'robustness.md'}  ({len(rows)} rows, {len(variants)} variants)")
    return out_dir
