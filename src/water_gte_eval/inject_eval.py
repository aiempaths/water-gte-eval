"""Inject demo driver: inject faults into clean telemetry, plot examples to
reports/, and run a severity sweep so detection is reported as a curve.

Pipeline per config: load clean tidy parquet -> apply each example injection
(plots of clean-vs-injected tags with the interval shaded -> reports/) ->
severity sweep: inject one fault at each severity, build windows, label with
the injected interval, score with a detector baseline fit on clean windows,
flag at a clean-score quantile, event-level detection recall per severity ->
reports/severity_curve.png + severity_curve.json + inject_demo.json.
Seeded and deterministic (plots/json aside, data outputs are deterministic).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from gte_eval.baselines import make_baseline
from gte_eval.engine_adapter import HistoryWindow, make_engine
from gte_eval.inject import inject_scenario, severity_sweep

from .data import TelemetryColumns, build_sample_windows, label_windows
from .metrics import build_event_records
from .report_metrics import rank_normalize_on_val


class InjectDemoConfigError(Exception):
    pass


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise InjectDemoConfigError(msg)


def load_inject_config(path: str | Path) -> dict:
    p = Path(path)
    if not p.is_file():
        raise InjectDemoConfigError(f"inject config not found: {p}")
    try:
        raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise InjectDemoConfigError(f"{p} is not valid YAML: {exc}") from exc
    _check(isinstance(raw, dict), "config root must be a mapping")
    _check(isinstance(raw.get("name"), str) and raw["name"], "'name' required")
    _check(isinstance(raw.get("seed"), int), "'seed' must be an int")
    _check(isinstance(raw.get("clean_path"), str), "clean_path required (parquet with DatetimeIndex)")
    _check(isinstance(raw.get("window"), dict), "window.length/stride required")
    _check(isinstance(raw.get("injections"), list), "injections must be a list")
    _check(isinstance(raw.get("sweep"), dict), "sweep section required")
    _check(isinstance(raw.get("output"), dict) and raw["output"].get("dir"), "output.dir required")
    return raw


def _resolve(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else Path.cwd() / p


def _dump(obj: dict) -> str:
    return json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n"


def _plot_injection(clean: pd.DataFrame, mod: pd.DataFrame, intervals: pd.DataFrame,
                    inj: dict, out_path: Path, max_tags: int = 4) -> None:
    row = intervals.iloc[0]
    tags = json.loads(row["tags"]) if row["tags"] != "all" else list(clean.columns)
    tags = [t for t in tags if t in clean.columns][:max_tags]
    fig, axes = plt.subplots(len(tags), 1, figsize=(10, 2.2 * len(tags)), sharex=True)
    if len(tags) == 1:
        axes = [axes]
    for ax, tag in zip(axes, tags):
        ax.plot(clean.index, clean[tag], lw=0.8, label="clean", color="tab:blue")
        ax.plot(mod.index, mod[tag], lw=0.8, label="injected", color="tab:red")
        ax.axvspan(row["start"], row["end"], color="orange", alpha=0.25,
                   label=f"{row['fault']} sev={row['severity']} label={row['label']}")
        ax.set_ylabel(tag)
        ax.legend(loc="upper right", fontsize=7)
    axes[-1].set_xlabel("time")
    fig.suptitle(f"injection {row['event_id']}: {row['fault']} (label={row['label']})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def run_inject_demo(config_path: str | Path) -> Path:
    cfg = load_inject_config(config_path)
    seed = int(cfg["seed"])
    clean_path = _resolve(cfg["clean_path"])
    if not clean_path.is_file():
        raise InjectDemoConfigError(
            f"clean_path not found: {clean_path}. Generate clean telemetry first, e.g. "
            "`python -m water_gte_eval sim-toy configs/sim/toy_example.yaml`."
        )
    clean = pd.read_parquet(clean_path)
    if not isinstance(clean.index, pd.DatetimeIndex):
        raise InjectDemoConfigError(f"{clean_path}: expected a DatetimeIndex")
    out_dir = _resolve(cfg["output"]["dir"])
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    tcols = TelemetryColumns(timestamp=clean.index.name or "timestamp",
                             sensors=tuple(clean.columns))
    wlen = int(cfg["window"]["length"])
    wstride = int(cfg["window"]["stride"])
    lab_cfg = cfg.get("labeling", {})
    min_frac = float(lab_cfg.get("min_overlap_frac", 0.5))
    lag = pd.Timedelta(lab_cfg.get("lag", "0s"))

    # 1. example injections -> plots + intervals parquet
    all_intervals = []
    for i, inj in enumerate(cfg["injections"]):
        inj = dict(inj)
        inj.setdefault("event_id", f"demo{i:03d}")
        mod, intervals = inject_scenario(clean, [inj], seed=seed + 1000 + i)
        all_intervals.append(intervals)
        fault = intervals.iloc[0]["fault"]
        eid = intervals.iloc[0]["event_id"]
        _plot_injection(clean, mod, intervals, inj,
                        out_dir / f"injected_{fault}_{eid}.png")
        print(f"injected {eid}: {fault} sev={intervals.iloc[0]['severity']} "
              f"label={intervals.iloc[0]['label']} -> {out_dir / f'injected_{fault}_{eid}.png'}")
    intervals_all = pd.concat(all_intervals, ignore_index=True) if all_intervals else pd.DataFrame(
        columns=["event_id", "start", "end", "label", "fault", "severity", "tags"])
    intervals_all.to_parquet(out_dir / "injected_intervals.parquet", index=False)

    # 2. severity sweep: baselines AND (optionally) the engine as detectors,
    #    all fit on clean windows; scores rank-normalized on the clean reference
    sw = cfg["sweep"]
    severities = [float(s) for s in sw["severities"]]
    det_names = list(sw.get("detectors") or ([sw["detector"]] if sw.get("detector") else ["simple_knn_trust"]))
    engine_cfg = sw.get("engine")  # optional {type: mock, params: {...}}
    q = float(sw.get("flag_quantile", 0.95))
    min_det = int(sw.get("min_detected_windows", 1))

    clean_windows = build_sample_windows(clean, tcols, wlen, wstride)
    clean_adapter = [w.to_adapter() for w in clean_windows]

    def make_scorer(name):
        if name == "engine":
            eng = make_engine(engine_cfg or {"type": "mock", "params": {}})
            eng.reset_history()
            eng.add_history([
                HistoryWindow(window_id=a.window_id, timestamps=list(a.timestamps),
                              sensors={k: list(v) for k, v in a.sensors.items()},
                              outcome_label="normal")
                for a in clean_adapter
            ])
            def score(ws):
                out = []
                for w in ws:
                    r = eng.score(w.to_adapter())
                    out.append(float(r.closeness) + (1.0 - float(r.agreement)))
                return out
            return score
        b = make_baseline({"name": name, "seed": seed,
                           "params": (sw.get("detector_params") or {}).get(name, {})})
        b.fit(clean_adapter, ["normal"] * len(clean_windows))
        return lambda ws: b.score_windows([w.to_adapter() for w in ws])

    all_names = list(det_names) + (["engine"] if engine_cfg else [])
    scorers = {name: make_scorer(name) for name in all_names}
    clean_raw = {name: np.array(scorers[name](clean_windows), dtype=float) for name in all_names}
    clean_norm = {name: rank_normalize_on_val(clean_raw[name], clean_raw[name]) for name in all_names}
    thresholds = {name: float(np.quantile(clean_norm[name], q)) for name in all_names}

    sweep = severity_sweep(clean, sw["fault"], sw["start"], sw["end"], severities,
                           tags=sw.get("tags"), label=sw.get("label"), seed=seed,
                           params=sw.get("params"))
    curves: dict[str, list] = {name: [] for name in all_names}
    for sev in severities:
        mod, intervals = sweep[sev]
        windows = build_sample_windows(mod, tcols, wlen, wstride)
        label_windows(windows, intervals, min_overlap_frac=min_frac, lag=lag)
        for name in all_names:
            raw = np.array(scorers[name](windows), dtype=float)
            scores = rank_normalize_on_val(clean_raw[name], raw)
            flags = {w.window_id: bool(s >= thresholds[name]) for w, s in zip(windows, scores)}
            records = build_event_records(windows, flags, intervals.iloc[0]["label"], min_det)
            detected = sum(1 for r in records if r.detected)
            curves[name].append({"severity": sev, "detected_events": detected,
                                 "n_events": len(records),
                                 "event_recall": detected / len(records) if records else 0.0,
                                 "n_windows_flagged": sum(flags.values()), "n_windows": len(windows)})
        print(f"sweep {sw['fault']} sev={sev}: " + "  ".join(
            f"{name} recall={curves[name][-1]['event_recall']:.2f}" for name in all_names))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name in all_names:
        ax.plot([c["severity"] for c in curves[name]], [c["event_recall"] for c in curves[name]],
                marker="o", label=name)
    ax.set_xlabel(f"severity of {sw['fault']}")
    ax.set_ylabel("event recall (detected injected intervals)")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f"detection curves — detectors fit on clean windows, threshold=q{q} of clean scores")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "severity_curve.png", dpi=110)
    plt.close(fig)

    summary = {
        "name": cfg["name"],
        "seed": seed,
        "clean_path": str(clean_path),
        "detectors": all_names,
        "flag_thresholds": thresholds,
        "flag_quantile": q,
        "window": {"length": wlen, "stride": wstride},
        "sweep": {"fault": sw["fault"], "start": str(sw["start"]), "end": str(sw["end"]),
                  "curve": curves[all_names[0]], "detectors": curves},
        "example_injections": all_intervals and pd.concat(all_intervals).to_dict("records"),
        "outputs": sorted(p.name for p in out_dir.iterdir()),
    }
    (out_dir / "inject_demo.json").write_text(_dump(summary), encoding="utf-8")
    (out_dir / "severity_curve.json").write_text(
        _dump({"fault": sw["fault"], "curve": curves[all_names[0]], "detectors": curves}),
        encoding="utf-8")
    print(f"report dir: {out_dir}")
    return out_dir
