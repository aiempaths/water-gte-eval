"""Report generator: builds reports/<run_hash>/report.md + PNG figures from a run
directory's results.parquet ALONE (no raw data needed). Optionally, if the report
config `windows_path` points at the prepare windows parquet, failure windows also
get raw tag plots.

Per method (engine__<mode> per history mode, baseline__<name> per baseline):
scores are rank-normalized on VAL (common val-referenced [0,1] scale; the engine's
combined score = ranknorm(closeness + (1 - agreement)), and closeness/agreement
are also reported separately as components). PRIMARY comparison = fixed review
budgets (5/10/20%) applied identically to engine and baselines, plus AUROC/AUPRC,
workload (escalations/hour), event detection fraction + delay, and fault vs
process_change discrimination — all with bootstrap CIs resampled over events
(bootstrap_n from config).

The TEST split is the only source of claims. A VAL section is included, clearly
marked "thresholds were fit here; NOT evidence". Ends with a plain-language
claims supported / not supported section that states the number of independent
events and refuses conclusions the sample size can't carry.
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
    rank_normalize_on_val,
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
    "include_val_section": True,
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
    if b is None or b.get("mean") is None:
        return "n/a"
    return f"{b['mean']:.4f} [{b['ci_low']:.4f}, {b['ci_high']:.4f}]"


def method_frame(res: pd.DataFrame, method: str, split: str, positive: set) -> pd.DataFrame:
    """Per-window frame for one method on one split: y (positive->1), score_raw,
    flag, plus engine closeness/agreement components for engine__ methods."""
    if method.startswith("engine__"):
        mode = method.split("__", 1)[1]
        m = res[(res["mode"] == mode) & (res["split"] == split)].drop_duplicates("window_id").copy()
    else:
        m = res[res["split"] == split].drop_duplicates("window_id").copy()
    m["y"] = m["true_label"].isin(positive).astype(int)
    if method.startswith("engine__"):
        m["closeness"] = m["engine_closeness"].astype(float)
        m["agreement"] = m["engine_agreement"].astype(float)
        m["score_raw"] = m["closeness"] + (1.0 - m["agreement"])
        m["flag"] = m["engine_flag"].astype(bool)
    else:
        m["score_raw"] = m[method].astype(float)
        m["flag"] = m[f"{method}__flag"].astype(bool)
    return m


def frame_metrics(mdf: pd.DataFrame, cfg: dict, t_min, t_max, do_bootstrap: bool = True) -> dict:
    """All report metrics for one method frame (expects a `score` column)."""
    scores, y = mdf["score"].to_numpy(), mdf["y"].to_numpy()
    budgets = [float(b) for b in cfg["budgets"]]
    min_det = int(cfg["min_detected_windows"])
    boot_n, ci, seed = int(cfg["bootstrap_n"]), float(cfg["ci"]), int(cfg["seed"])
    m = {
        "auroc": auroc(y, scores),
        "auprc": auprc(y, scores),
        "false_trust_at_budget": {b: false_trust_at_budget(scores, y, b) for b in budgets},
        "workload_escalations_per_hour": escalations_per_hour(mdf["flag"].to_numpy(), t_min, t_max),
        "risk_coverage": risk_coverage_points(scores, y, n_points=50),
    }
    _, agg = event_detection(mdf, min_det)
    m["events"] = agg
    m["discrimination"] = discrimination(mdf, cfg["discrimination_labels"])
    if not do_bootstrap:
        m["bootstrap"] = {"note": "bootstrap only on the claims split (test)"}
        return m

    def fn_auroc(s): return auroc(s["y"].to_numpy(), s["score"].to_numpy())
    def fn_auprc(s): return auprc(s["y"].to_numpy(), s["score"].to_numpy())
    def fn_ftr(s): return false_trust_at_budget(s["score"].to_numpy(), s["y"].to_numpy(), budgets[0])["false_trust_rate"]
    def fn_det(s): return event_detection(s, min_det)[1]["fraction_detected"]
    def fn_work(s): return escalations_per_hour(s["flag"].to_numpy(), s["start"].min(), s["end"].max())
    m["bootstrap"] = {
        "auroc": bootstrap_over_events(mdf, fn_auroc, boot_n, seed, ci),
        "auprc": bootstrap_over_events(mdf, fn_auprc, boot_n, seed, ci),
        f"false_trust_rate@{budgets[0]}": bootstrap_over_events(mdf, fn_ftr, boot_n, seed, ci),
        "event_fraction_detected": bootstrap_over_events(mdf, fn_det, boot_n, seed, ci),
        "escalations_per_hour": bootstrap_over_events(mdf, fn_work, boot_n, seed, ci),
    }
    return m


def summarize_run_results(run_dir: str | Path, positive_labels, budgets, bootstrap_n: int,
                          ci: float, seed: int, split: str = "test",
                          min_detected_windows: int = 1,
                          discrimination_labels=("fault", "process_change")) -> dict:
    """Per-method metrics (with event-bootstrap CIs) for one run dir's
    results.parquet. Used by the sweep and robustness drivers so they share the
    exact metric code of the report. Scores are rank-normalized on val, as in
    build_report."""
    run_dir = _resolve(run_dir)
    res = pd.read_parquet(run_dir / "results.parquet")
    positive = set(positive_labels)
    methods = _methods(res)
    cfg = {"budgets": [float(b) for b in budgets], "bootstrap_n": int(bootstrap_n),
           "ci": float(ci), "seed": int(seed), "min_detected_windows": int(min_detected_windows),
           "discrimination_labels": list(discrimination_labels)}
    df_split = res[res["split"] == split]
    t_min, t_max = df_split["start"].min(), df_split["end"].max()
    out: dict[str, dict] = {}
    for method in methods:
        val_frame = method_frame(res, method, "val", positive)
        m_frame = method_frame(res, method, split, positive).copy()
        m_frame["score"] = rank_normalize_on_val(val_frame["score_raw"].to_numpy(),
                                                 m_frame["score_raw"].to_numpy())
        out[method] = frame_metrics(m_frame, cfg, t_min, t_max, do_bootstrap=True)
    return {"run_dir": str(run_dir), "split": split, "methods": out,
            "n_events": int(df_split.loc[df_split["event_id"].astype(str) != "", "event_id"].nunique())}


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
    if split not in set(res["split"].unique()):
        raise ReportConfigError(f"split {split!r} not in results.parquet (have {sorted(set(res['split']))})")
    positive = set(cfg["positive_labels"])
    methods = _methods(res)
    budgets = [float(b) for b in cfg["budgets"]]

    df_test = res[res["split"] == split]
    t_min, t_max = df_test["start"].min(), df_test["end"].max()
    eval_hours = (pd.Timestamp(t_max) - pd.Timestamp(t_min)).total_seconds() / 3600.0
    n_events = int(df_test.loc[df_test["event_id"].astype(str) != "", "event_id"].nunique())

    all_metrics: dict[str, dict] = {}
    for method in methods:
        val_frame = method_frame(res, method, "val", positive)
        test_frame = method_frame(res, method, split, positive)
        # rank-normalize on val: common [0,1] val-referenced scale for every method
        test_frame = test_frame.copy()
        test_frame["score"] = rank_normalize_on_val(val_frame["score_raw"].to_numpy(),
                                                    test_frame["score_raw"].to_numpy())
        val_frame = val_frame.copy()
        val_frame["score"] = val_frame["score_raw"]
        m_test = frame_metrics(test_frame, cfg, t_min, t_max)
        entry = dict(m_test)
        if method.startswith("engine__"):
            y = test_frame["y"].to_numpy()
            entry["components"] = {
                "closeness_auroc": auroc(y, test_frame["closeness"].to_numpy()),
                "one_minus_agreement_auroc": auroc(y, (1.0 - test_frame["agreement"]).to_numpy()),
                "combined_ranknorm_auroc": m_test["auroc"],
            }
        if cfg.get("include_val_section", True) and len(val_frame):
            # val is NOT evidence: point metrics only, no bootstrap CIs there
            entry["val"] = frame_metrics(val_frame, cfg, val_frame["start"].min(),
                                            val_frame["end"].max(), do_bootstrap=False)
        all_metrics[method] = entry

        # risk-coverage figure (test split)
        pts = m_test["risk_coverage"]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker=".", ms=3)
        ax.set_xlabel("coverage (fraction of windows routed to humans, least-trusted first)")
        ax.set_ylabel("error rate among trusted windows (false-trust rate)")
        ax.set_title(f"risk-coverage — {method} ({split}, scores rank-normalized on val)")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / f"risk_coverage_{method}.png", dpi=110)
        plt.close(fig)

    # overview figure: AUROC / AUPRC per method with bootstrap CI (test)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key, title in ((axes[0], "auroc", "AUROC"), (axes[1], "auprc", "AUPRC")):
        vals = [all_metrics[m][key] for m in methods]
        bs = [all_metrics[m]["bootstrap"][key] for m in methods]
        yerr = [[(v - b["ci_low"]) if (v is not None and b["mean"] is not None) else 0 for v, b in zip(vals, bs)],
                [(b["ci_high"] - v) if (v is not None and b["mean"] is not None) else 0 for v, b in zip(vals, bs)]]
        ax.bar(methods, [v if v is not None else 0 for v in vals], yerr=yerr, capsize=3)
        ax.axhline(0.5 if key == "auroc" else float(df_test["true_label"].isin(positive).mean()),
                   ls="--", c="gray", lw=0.8)
        ax.set_title(f"{title} ({split}, bootstrap CI over {n_events} events)")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / "metrics_overview.png", dpi=110)
    plt.close(fig)

    # discrimination figure (test)
    disc_labels = cfg["discrimination_labels"]
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(methods))
    w = 0.35
    for j, lab in enumerate(disc_labels):
        vals = []
        for mname in methods:
            d = all_metrics[mname]["discrimination"]["event_detection_by_label"].get(lab, {})
            v = d.get("fraction_detected")
            vals.append(v if v is not None else 0.0)
        ax.bar(x + (j - 0.5) * w, vals, width=w, label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha="right")
    ax.set_ylabel("event detection rate")
    ax.set_title(f"discrimination: {' vs '.join(disc_labels)} events ({split})")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / "discrimination.png", dpi=110)
    plt.close(fig)

    # failures first (engine frozen if present, else first engine mode)
    engine_modes = [m for m in methods if m.startswith("engine__")]
    fail_method = "engine__frozen" if "engine__frozen" in engine_modes else (engine_modes[0] if engine_modes else methods[0])

    # ablation: engine vs simple_knn_trust side by side, plus a paired bootstrap CI
    # on the AUROC difference (events resampled jointly, so the CI is paired)
    ablation = None
    skt = "baseline__simple_knn_trust"
    if engine_modes and skt in methods:
        engine_type = None
        manifest_path = run_dir / "manifest.json"
        if manifest_path.is_file():
            engine_type = (json.loads(manifest_path.read_text(encoding="utf-8")).get("engine") or {}).get("type")
        ref_engine = "engine__frozen" if "engine__frozen" in engine_modes else engine_modes[0]
        ef = method_frame(res, ref_engine, split, positive).copy()
        ef["score"] = rank_normalize_on_val(method_frame(res, ref_engine, "val", positive)["score_raw"].to_numpy(),
                                             ef["score_raw"].to_numpy())
        sf = method_frame(res, skt, split, positive).copy()
        sf["score"] = rank_normalize_on_val(method_frame(res, skt, "val", positive)["score_raw"].to_numpy(),
                                            sf["score_raw"].to_numpy())
        both = ef[["window_id", "y", "event_id", "score"]].merge(
            sf[["window_id", "event_id", "score"]], on=["window_id", "event_id"],
            suffixes=("_e", "_s"))

        def fn_delta(s):
            a = auroc(s["y"].to_numpy(), s["score_e"].to_numpy())
            b = auroc(s["y"].to_numpy(), s["score_s"].to_numpy())
            return None if a is None or b is None else a - b

        ablation = {
            "engine_type": engine_type,
            "engine_method": ref_engine,
            "baseline": skt,
            "delta_auroc": bootstrap_over_events(both, fn_delta, int(cfg["bootstrap_n"]),
                                                 int(cfg["seed"]), float(cfg["ci"])),
            "n_events": n_events,
        }
    ff = method_frame(res, fail_method, split, positive)
    vf = method_frame(res, fail_method, "val", positive)
    ff = ff.copy()
    ff["score"] = rank_normalize_on_val(vf["score_raw"].to_numpy(), ff["score_raw"].to_numpy())
    failures = ff[(ff["flag"] == False) & (ff["y"] == 1)].sort_values("score").head(int(cfg["failures_top_k"]))

    fig, ax = plt.subplots(figsize=(8, 4))
    if len(failures):
        ax.barh(failures["window_id"].astype(str), failures["score"].astype(float), color="tab:red")
        ax.set_xlabel(f"rank-normalized untrust score of the wrongly-trusted window ({fail_method})")
        ax.set_title(f"failures first: top {len(failures)} highest-confidence wrong trusts ({split})")
    else:
        ax.text(0.5, 0.5, f"no wrong trusts on {split}", ha="center", va="center", transform=ax.transAxes)
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(out_dir / "failures_first.png", dpi=110)
    plt.close(fig)

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
    lines.append(f"- claims split: **{split}** — windows: {len(df_test.drop_duplicates('window_id'))}  "
                 f"independent events: **{n_events}**  eval span: {t_min} .. {t_max} ({eval_hours:.1f} h)")
    lines.append(f"- positive (anomalous/wrong) labels: {cfg['positive_labels']}")
    lines.append(f"- scores rank-normalized on the VAL split (common [0,1] scale); bootstrap: "
                 f"{cfg['bootstrap_n']} resamples over events/scenarios, CI {cfg['ci']}")
    lines.append("- **PRIMARY comparison: fixed review budgets "
                 + "/".join(f"{b:.0%}" for b in budgets)
                 + " applied identically to the engine and all baselines** (rank-based, threshold-free). "
                 "Val-fit F1 thresholds are secondary (see metrics.json).")
    lines.append("")
    lines.append(f"## Per-method metrics — {split} split (the only source of claims)")
    lines.append("")
    lines.append("| method | AUROC | AUPRC | false-trust @ " +
                 "/".join(f"{b:.0%}" for b in budgets) + " | escalations/h | events detected | mean delay (h) |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for mname in methods:
        m = all_metrics[mname]
        ftr = [m["false_trust_at_budget"][b]["false_trust_rate"] for b in budgets]
        lines.append(
            f"| {mname} | {_fmt(m['auroc'])} ({_ci_str(m['bootstrap']['auroc'])}) "
            f"| {_fmt(m['auprc'])} ({_ci_str(m['bootstrap']['auprc'])}) "
            f"| {'/'.join(_fmt(v, 3) for v in ftr)} ({_ci_str(m['bootstrap'][f'false_trust_rate@{budgets[0]}'])}) "
            f"| {_fmt(m['workload_escalations_per_hour'], 2)} ({_ci_str(m['bootstrap']['escalations_per_hour'])}) "
            f"| {_fmt(m['events']['fraction_detected'], 3)} ({_ci_str(m['bootstrap']['event_fraction_detected'])}) "
            f"| {_fmt(m['events']['mean_delay_hours'], 2)} |")
    lines.append("")
    eng = [m for m in methods if m.startswith("engine__")]
    if eng:
        lines.append("### Engine score components (AUROC on " + split + "): closeness vs agreement vs combined (rank-norm on val)")
        lines.append("")
        lines.append("| method | closeness AUROC | (1-agreement) AUROC | combined AUROC |")
        lines.append("| --- | --- | --- | --- |")
        for mname in eng:
            c = all_metrics[mname]["components"]
            lines.append(f"| {mname} | {_fmt(c['closeness_auroc'])} | {_fmt(c['one_minus_agreement_auroc'])} "
                         f"| {_fmt(c['combined_ranknorm_auroc'])} |")
        lines.append("")
    if ablation:
        lines.append(f"## Ablation: engine vs simple_knn_trust ({split}, event-bootstrap CIs)")
        lines.append("")
        if ablation["engine_type"] == "mock":
            lines.append("**MOCK ENGINE \u2014 the numbers in this section are NOT evidence about a real GTE.**")
            lines.append("")
        lines.append("| method | AUROC (CI) | AUPRC (CI) | events detected (CI) | false-trust @ "
                     + "/".join(f"{b:.0%}" for b in budgets) + " |")
        lines.append("| --- | --- | --- | --- | --- |")
        for mname in (ablation["engine_method"], ablation["baseline"]):
            m = all_metrics[mname]
            ftr = [m["false_trust_at_budget"][b]["false_trust_rate"] for b in budgets]
            lines.append(f"| {mname} | {_fmt(m['auroc'])} ({_ci_str(m['bootstrap']['auroc'])}) "
                         f"| {_fmt(m['auprc'])} ({_ci_str(m['bootstrap']['auprc'])}) "
                         f"| {_fmt(m['events']['fraction_detected'], 3)} ({_ci_str(m['bootstrap']['event_fraction_detected'])}) "
                         f"| {'/'.join(_fmt(v, 3) for v in ftr)} |")
        lines.append("")
        lines.append(f"\u0394AUROC (engine \u2212 simple_knn_trust), paired bootstrap over {ablation['n_events']} "
                     f"events: {_ci_str(ablation['delta_auroc'])}")
        lines.append("")
    lines.append("Figures: `metrics_overview.png` (AUROC/AUPRC + CI), `risk_coverage_<method>.png` per method, "
                 "`discrimination.png`, `failures_first.png`"
                 + (", `failure_<window_id>.png` per failure window" if failure_plots else "") + ".")
    lines.append("")

    if cfg.get("include_val_section", True):
        lines.append("## Val split — thresholds were fit here; NOT evidence")
        lines.append("")
        lines.append("Metrics on val are reported for transparency only. The val split was used to fit "
                     "thresholds / normalization references; no claims are drawn from it.")
        lines.append("")
        lines.append("| method | AUROC | AUPRC | false-trust @ " +
                     "/".join(f"{b:.0%}" for b in budgets) + " | escalations/h | events detected |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for mname in methods:
            if "val" not in all_metrics[mname]:
                continue
            m = all_metrics[mname]["val"]
            ftr = [m["false_trust_at_budget"][b]["false_trust_rate"] for b in budgets]
            lines.append(f"| {mname} | {_fmt(m['auroc'])} | {_fmt(m['auprc'])} "
                         f"| {'/'.join(_fmt(v, 3) for v in ftr)} "
                         f"| {_fmt(m['workload_escalations_per_hour'], 2)} "
                         f"| {_fmt(m['events']['fraction_detected'], 3)} |")
        lines.append("")

    lines.append("## Discrimination: " + " vs ".join(disc_labels) + f" ({split})")
    lines.append("")
    lines.append("| method | " + " event detection | ".join(disc_labels) +
                 " | discrimination gap | window AUROC |")
    lines.append("| --- | --- | --- | --- | --- |")
    for mname in methods:
        d = all_metrics[mname]["discrimination"]
        ed = d["event_detection_by_label"]
        lines.append(f"| {mname} | {_fmt(ed.get(disc_labels[0], {}).get('fraction_detected'), 3)} "
                     f"| {_fmt(ed.get(disc_labels[1], {}).get('fraction_detected'), 3)} "
                     f"| {_fmt(d['discrimination_gap'], 3)} | {_fmt(d['window_auroc_fault_vs_process_change'])} |")
    lines.append("")
    lines.append("## Failures first (highest-confidence wrong “trust” windows)")
    lines.append("")
    if len(failures):
        lines.append(f"Method: `{fail_method}`; a window is “trusted” when not flagged. "
                     f"Top {len(failures)} trusted-but-anomalous windows on {split}, most confident first:")
        lines.append("")
        lines.append("| window_id | event_id | true label | rank-norm untrust | neighbors |")
        lines.append("| --- | --- | --- | --- | --- |")
        for _, row in failures.iterrows():
            neigh = row.get("engine_neighbors", "n/a") if fail_method.startswith("engine__") else "n/a"
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

    # ---------------- claims (test split only) ----------------
    lines.append("## Claims supported / not supported")
    lines.append("")
    lines.append(f"Claims below use ONLY the **{split}** split: **{n_events} independent event(s)** "
                 f"({len(df_test.drop_duplicates('window_id'))} windows). "
                 "Bootstrap CIs resample events, not windows, because windows within an event are dependent. "
                 "The val split was used for thresholds/normalization and is not evidence.")
    lines.append("")
    min_events = int(cfg["min_events_for_claims"])
    lines.append("**Supported:**")
    any_supported = False
    if n_events >= min_events:
        for mname in methods:
            b = all_metrics[mname]["bootstrap"]["auroc"]
            if b["mean"] is not None and b["ci_low"] is not None and b["ci_low"] > 0.5:
                lines.append(f"- {mname} ranks anomalous windows above normal ones "
                             f"(AUROC {b['mean']:.3f}, CI [{b['ci_low']:.3f}, {b['ci_high']:.3f}] excludes 0.5) — {n_events} events.")
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
        lines.append(f"- Any comparison between methods or 'engine beats baseline X' claims: "
                     f"only {n_events} independent event(s) < {min_events} required; CIs shown but not interpreted.")
    lines.append("- Any claim from the val split (thresholds were fit there).")
    lines.append("- Any claim about individual windows or single scenarios as evidence; windows are not independent samples.")
    lines.append("- Any claim that process_change windows should be flagged: they are legitimate disturbances; "
                 "see the discrimination table instead of counting them as detections.")
    lines.append("- Any generalization beyond the evaluated split/time range and the datasets in the run manifest.")
    lines.append("")

    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    (out_dir / "metrics.json").write_text(
        json.dumps({"run_hash": run_hash, "split": split, "n_events": n_events,
                    "methods": all_metrics},
                   indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(f"report: {out_dir / 'report.md'}  ({len(methods)} methods, {n_events} events on {split})")
    return out_dir
