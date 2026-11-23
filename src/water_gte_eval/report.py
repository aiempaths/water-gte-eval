"""Report generator: builds reports/<run_hash>/report.md + PNG figures from a run
directory's results.parquet ALONE (no raw data needed). Optionally, if
report config `windows_path` points at the prepare windows parquet, failure
windows also get raw tag plots.

Per method (engine__<mode> for each history mode, baseline__<name> for each
baseline) on the configured split (default test): AUROC/AUPRC, false-trust rate at
review budgets, risk-coverage curve (figure), workload (escalations/hour),
event-level detection fraction + delay, discrimination fault vs process_change.
All with bootstrap CIs resampled over events (bootstrap_n from config). Ends with
a plain-language "claims supported / not supported" section that states the
number of independent events and refuses conclusions the sample size can't carry.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from .report_metrics import (
    auroc,
    auprc,
    bootstrap_over_events,
    discrimination,
    escalations_per_hour,
    event_detection,
    false_trust_at_budget,
    method_scores,
    risk_coverage_points,
)

DEFAULTS = {
    "bootstrap_n": 1000,
    "ci": 0.95,
    "seed": 42,
    "budgets": [0.05, 0.10, 0.20],
    "positive_labels": ["attack", "fault"],
    "discrimination_labels": ["fault", "process_change"],
    "split": "test",
    "min_detected_windows": 1,
    "failures_top_k": 5,
    "min_events_for_claims": 5,
    "windows_path": None,
    "output_dir": "reports",
}


class ReportConfigError(Exception):
    pass


def load_report_config(path: str | Path | None) -> dict:
    cfg = dict(DEFAULTS)
    if path:
        p = Path(path)
        if not p.is_file():
            raise ReportConfigError(f"report config not found: {p}")
        cfg.update(yaml.safe_load(p.read_text(encoding="utf-8")) or {})
    return cfg


def _resolve(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else Path.cwd() / p


def _methods(df: pd.DataFrame) -> list[str]:
    methods = [f"engine__{m}" for m in sorted(df["mode"].unique())]
    methods += sorted(c for c in df.columns if c.startswith("baseline__") and not c.endswith("__flag"))
    return methods


def _fmt(x, nd=4):
    return "n/a" if x is None else f"{x:.{nd}f}"


def _ci_str(b):
    if b is None or b["mean"] is None:
        return "n/a"
    return f"{b['mean']:.4f} [{b['ci_low']:.4f}, {b['ci_high']:.4f}]"


def build_report(run_dir: str | Path, config_path: str | Path | None = None) -> Path:
    run_dir = _resolve(run_dir)
    results_path = run_dir / "results.parquet"
    if not results_path.is_file():
        raise ReportConfigError(
            f"results.parquet not found in {run_dir}. Run `eval` first "
            "(python -m water_gte_eval eval <config>)."
        )
    cfg = load_report_config(config_path)
    res = pd.read_parquet(results_path)
    run_hash = str(res["config_hash"].iloc[0])
    out_dir = _resolve(cfg["output_dir"]) / run_hash
    out_dir.mkdir(parents=True, exist_ok=True)

    split = cfg["split"]
    df = res[res["split"] == split].copy()
    if df.empty:
        raise ReportConfigError(f"no rows with split={split!r} in {results_path}")
    df["y"] = df["true_label"].isin(cfg["positive_labels"]).astype(int)
    # one row per window per method
    methods = _methods(df)
    budgets = [float(b) for b in cfg["budgets"]]
    min_det = int(cfg["min_detected_windows"])
    boot_n = int(cfg["bootstrap_n"])
    ci = float(cfg["ci"])
    seed = int(cfg["seed"])

    t_min, t_max = df["start"].min(), df["end"].max()
    eval_hours = (pd.Timestamp(t_max) - pd.Timestamp(t_min)).total_seconds() / 3600.0
    n_events = int(df.loc[df["event_id"].astype(str) != "", "event_id"].nunique())

    all_metrics: dict[str, dict] = {}
    for method in methods:
        mdf = df.drop_duplicates("window_id")  # same window per mode row; baselines repeat per mode
        mdf = mdf.copy()
        if method.startswith("engine__"):
            mode = method.split("__", 1)[1]
            mdf = df[(df["mode"] == mode)].drop_duplicates("window_id").copy()
        scores, y = method_scores(mdf, method)
        mdf = mdf.assign(score=scores)
        flags = (mdf["engine_flag"].astype(bool) if method.startswith("engine__")
                 else mdf[f"{method}__flag"].astype(bool))
        mdf = mdf.assign(flag=flags.to_numpy())

        m = {
            "auroc": auroc(y, scores),
            "auprc": auprc(y, scores),
            "false_trust_at_budget": {b: false_trust_at_budget(scores, y, b) for b in budgets},
            "workload_escalations_per_hour": escalations_per_hour(flags, t_min, t_max),
            "risk_coverage": risk_coverage_points(scores, y, n_points=50),
        }
        per_event, agg = event_detection(mdf, min_det)
        m["events"] = agg
        m["events_detail"] = per_event
        m["discrimination"] = discrimination(mdf, cfg["discrimination_labels"])

        # bootstrap over events for the scalar metrics
        def fn_auroc(s): return auroc(s["y"].to_numpy(), s["score"].to_numpy())
        def fn_auprc(s): return auprc(s["y"].to_numpy(), s["score"].to_numpy())
        def fn_ftr(s, b=budgets[0]): return false_trust_at_budget(s["score"].to_numpy(), s["y"].to_numpy(), b)["false_trust_rate"]
        def fn_det(s): return event_detection(s, min_det)[1]["fraction_detected"]
        def fn_work(s): return escalations_per_hour(s["flag"].to_numpy(), s["start"].min(), s["end"].max())
        m["bootstrap"] = {
            "auroc": bootstrap_over_events(mdf, fn_auroc, boot_n, seed, ci),
            "auprc": bootstrap_over_events(mdf, fn_auprc, boot_n, seed, ci),
            f"false_trust_rate@{budgets[0]}": bootstrap_over_events(mdf, fn_ftr, boot_n, seed, ci),
            "event_fraction_detected": bootstrap_over_events(mdf, fn_det, boot_n, seed, ci),
            "escalations_per_hour": bootstrap_over_events(mdf, fn_work, boot_n, seed, ci),
        }
        all_metrics[method] = m

        # risk-coverage figure
        pts = m["risk_coverage"]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker=".", ms=3)
        ax.set_xlabel("coverage (fraction of windows routed to humans, least-trusted first)")
        ax.set_ylabel("error rate among trusted windows (false-trust rate)")
        ax.set_title(f"risk-coverage — {method} ({split})")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / f"risk_coverage_{method}.png", dpi=110)
        plt.close(fig)

    # overview figure: AUROC / AUPRC per method with bootstrap CI
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    names = methods
    for ax, key, title in ((axes[0], "auroc", "AUROC"), (axes[1], "auprc", "AUPRC")):
        vals = [all_metrics[m][key] for m in names]
        los = [all_metrics[m]["bootstrap"][key]["ci_low"] if all_metrics[m]["bootstrap"][key]["mean"] is not None else 0 for m in names]
        his = [all_metrics[m]["bootstrap"][key]["ci_high"] if all_metrics[m]["bootstrap"][key]["mean"] is not None else 0 for m in names]
        yerr = [[(v - lo) if v is not None else 0 for v, lo in zip(vals, los)],
                [(hi - v) if v is not None else 0 for v, hi in zip(vals, his)]]
        ax.bar(names, [v if v is not None else 0 for v in vals], yerr=yerr, capsize=3)
        ax.axhline(0.5 if key == "auroc" else (df["y"].mean()), ls="--", c="gray", lw=0.8)
        ax.set_title(f"{title} ({split}, bootstrap CI over {n_events} events)")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / "metrics_overview.png", dpi=110)
    plt.close(fig)

    # discrimination figure: event detection by label per method
    disc_labels = cfg["discrimination_labels"]
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(methods))
    w = 0.35
    for j, lab in enumerate(disc_labels):
        vals = []
        for mname in methods:
            d = all_metrics[mname]["discrimination"]["event_detection_by_label"].get(lab, {})
            vals.append(d.get("fraction_detected") if d.get("fraction_detected") is not None else 0.0)
        ax.bar(x + (j - 0.5) * w, vals, width=w, label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha="right")
    ax.set_ylabel("event detection rate")
    ax.set_title("discrimination: fault vs process_change events")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / "discrimination.png", dpi=110)
    plt.close(fig)

    # failures first: highest-confidence wrong "trust" windows (engine frozen mode
    # if present, else first engine mode), ranked by lowest untrust score
    engine_modes = [m for m in methods if m.startswith("engine__")]
    fail_method = "engine__frozen" if "engine__frozen" in engine_modes else (engine_modes[0] if engine_modes else methods[0])
    fdf = df[df["mode"] == fail_method.split("__", 1)[1]].drop_duplicates("window_id").copy() if fail_method.startswith("engine__") else df.drop_duplicates("window_id").copy()
    scores, y = method_scores(fdf, fail_method)
    fdf = fdf.assign(score=scores)
    flags = (fdf["engine_flag"].astype(bool) if fail_method.startswith("engine__") else fdf[f"{fail_method}__flag"].astype(bool))
    fdf = fdf.assign(flag=flags.to_numpy())
    failures = fdf[(fdf["flag"] == False) & (fdf["y"] == 1)].sort_values("score").head(int(cfg["failures_top_k"]))

    fig, ax = plt.subplots(figsize=(8, 4))
    if len(failures):
        ax.barh(failures["window_id"].astype(str), failures["score"].astype(float), color="tab:red")
        ax.set_xlabel(f"untrust score of the wrongly-trusted window ({fail_method})")
        ax.set_title(f"failures first: top {len(failures)} highest-confidence wrong trusts")
    else:
        ax.text(0.5, 0.5, "no wrong trusts on " + split, ha="center", va="center", transform=ax.transAxes)
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(out_dir / "failures_first.png", dpi=110)
    plt.close(fig)

    # optional raw tag plots for failure windows (needs windows parquet; report
    # itself builds from results.parquet alone)
    failure_plots = []
    if cfg.get("windows_path") and len(failures):
        wdf = pd.read_parquet(_resolve(cfg["windows_path"]))
        tag_cols = [c for c in wdf.columns if c not in
                    ("window_id", "start", "end", "stage", "label", "event_id", "tag_group", "source_file")]
        for _, row in failures.iterrows():
            wrow = wdf[wdf["window_id"] == row["window_id"]]
            if wrow.empty:
                continue
            wrow = wrow.iloc[0]
            fig, axes = plt.subplots(len(tag_cols), 1, figsize=(9, 1.8 * len(tag_cols)), sharex=True)
            if len(tag_cols) == 1:
                axes = [axes]
            for ax, tag in zip(axes, tag_cols):
                arr = np.asarray(wrow[tag], dtype=float)
                ax.plot(np.arange(len(arr)), arr, lw=1)
                ax.set_ylabel(tag)
            axes[-1].set_xlabel("sample within window")
            axes[-1].set_title(f"failure window {row['window_id']} (label={row['true_label']}, "
                               f"event={row['event_id']}, score={row['score']:.3f})")
            fig.tight_layout()
            p = out_dir / f"failure_{row['window_id']}.png"
            fig.savefig(p, dpi=100)
            plt.close(fig)
            failure_plots.append(p.name)

    # ---------------- report.md ----------------
    lines = []
    lines.append(f"# Evaluation report — run `{run_hash}`")
    lines.append("")
    lines.append(f"- split reported: **{split}**  windows: {len(df.drop_duplicates('window_id'))}  "
                 f"independent events: **{n_events}**  eval span: {t_min} .. {t_max} ({eval_hours:.1f} h)")
    lines.append(f"- positive (anomalous/wrong) labels: {cfg['positive_labels']}")
    lines.append(f"- bootstrap: {boot_n} resamples over events/scenarios, CI {ci}")
    lines.append("")
    lines.append("## Per-method metrics (point estimate, bootstrap CI over events)")
    lines.append("")
    lines.append("| method | AUROC | AUPRC | false-trust @ " +
                 "/".join(f"{b:.0%}" for b in budgets) + " | escalations/h | events detected | mean delay (h) |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for mname in methods:
        m = all_metrics[mname]
        ftr = [m["false_trust_at_budget"][b]["false_trust_rate"] for b in budgets]
        ftr_s = "/".join(_fmt(v, 3) for v in ftr)
        lines.append(
            f"| {mname} | {_fmt(m['auroc'])} ({_ci_str(m['bootstrap']['auroc'])}) "
            f"| {_fmt(m['auprc'])} ({_ci_str(m['bootstrap']['auprc'])}) "
            f"| {ftr_s} ({_ci_str(m['bootstrap'][f'false_trust_rate@{budgets[0]}'])}) "
            f"| {_fmt(m['workload_escalations_per_hour'], 2)} ({_ci_str(m['bootstrap']['escalations_per_hour'])}) "
            f"| {_fmt(m['events']['fraction_detected'], 3)} ({_ci_str(m['bootstrap']['event_fraction_detected'])}) "
            f"| {_fmt(m['events']['mean_delay_hours'], 2)} |")
    lines.append("")
    lines.append("Figures: `metrics_overview.png` (AUROC/AUPRC + CI), `risk_coverage_<method>.png` per method, "
                 "`discrimination.png`, `failures_first.png`"
                 + (", `failure_<window_id>.png` per failure window" if failure_plots else "") + ".")
    lines.append("")
    lines.append("## Discrimination: fault vs legitimate process_change")
    lines.append("")
    lines.append("| method | fault event detection | process_change event detection | discrimination gap | window AUROC (fault vs p.c.) |")
    lines.append("| --- | --- | --- | --- | --- |")
    for mname in methods:
        d = all_metrics[mname]["discrimination"]
        ed = d["event_detection_by_label"]
        lines.append(f"| {mname} | {_fmt(ed.get('fault', {}).get('fraction_detected'), 3)} "
                     f"| {_fmt(ed.get('process_change', {}).get('fraction_detected'), 3)} "
                     f"| {_fmt(d['discrimination_gap'], 3)} | {_fmt(d['window_auroc_fault_vs_process_change'])} |")
    lines.append("")
    lines.append("## Failures first (highest-confidence wrong “trust” windows)")
    lines.append("")
    if len(failures):
        lines.append(f"Method: `{fail_method}`; a window is “trusted” when not flagged. "
                     f"Top {len(failures)} trusted-but-anomalous windows, most confident first:")
        lines.append("")
        lines.append("| window_id | event_id | true label | untrust score | neighbors |")
        lines.append("| --- | --- | --- | --- | --- |")
        for _, row in failures.iterrows():
            neigh = row["engine_neighbors"] if fail_method.startswith("engine__") else "n/a"
            if isinstance(neigh, (list, np.ndarray)):
                neigh = ", ".join(map(str, neigh))[:80]
            lines.append(f"| {row['window_id']} | {row['event_id']} | {row['true_label']} "
                         f"| {row['score']:.4f} | {neigh} |")
        if failure_plots:
            lines.append("")
            lines.append("Data plots: " + ", ".join(f"`{p}`" for p in failure_plots))
    else:
        lines.append(f"No wrong trusts on the {split} split for {fail_method}.")
    lines.append("")

    # ---------------- claims ----------------
    lines.append("## Claims supported / not supported")
    lines.append("")
    lines.append(f"This report is based on **{n_events} independent event(s)** "
                 f"({len(df.drop_duplicates('window_id'))} windows on the {split} split). "
                 "Bootstrap CIs resample events, not windows, because windows within an event are dependent.")
    lines.append("")
    min_events = int(cfg["min_events_for_claims"])
    lines.append("**Supported:**")
    any_supported = False
    if n_events >= min_events:
        for mname in methods:
            b = all_metrics[mname]["bootstrap"]["auroc"]
            if b["mean"] is not None and b["ci_low"] is not None and b["ci_low"] > 0.5:
                lines.append(f"- {mname} ranks anomalous windows above normal ones "
                             f"(AUROC {b['mean']:.3f}, 95% CI [{b['ci_low']:.3f}, {b['ci_high']:.3f}] excludes 0.5) — {n_events} events.")
                any_supported = True
            d = all_metrics[mname]["bootstrap"]["event_fraction_detected"]
            if d["mean"] is not None and d["ci_low"] is not None and d["ci_low"] > 0:
                lines.append(f"- {mname} detects a nonzero fraction of events "
                             f"({d['mean']:.3f}, CI [{d['ci_low']:.3f}, {d['ci_high']:.3f}]) — {n_events} events.")
                any_supported = True
    if not any_supported:
        lines.append(f"- None of the per-method ranking/detection claims meet the bar "
                     f"(need >= {min_events} independent events with a CI excluding the null).")
    lines.append("")
    lines.append("**Not supported (refused):**")
    if n_events < min_events:
        lines.append(f"- Any comparison between methods or any 'the engine beats baseline X' claim: "
                     f"only {n_events} independent event(s) < {min_events} required; CIs are shown but not interpreted.")
    lines.append("- Any claim about individual windows or single scenarios as evidence; windows are not independent samples.")
    lines.append("- Any claim that process_change windows should be flagged: they are legitimate disturbances; "
                 "see the discrimination table instead of counting them as detections.")
    lines.append("- Any generalization beyond the evaluated split/time range and the datasets in the run manifest.")
    lines.append("")

    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    (out_dir / "metrics.json").write_text(
        json.dumps({"run_hash": run_hash, "split": split, "n_events": n_events,
                    "methods": {k: {kk: vv for kk, vv in v.items() if kk != "events_detail"}
                                for k, v in all_metrics.items()}},
                   indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(f"report: {out_dir / 'report.md'}  ({len(methods)} methods, {n_events} events)")
    return out_dir
