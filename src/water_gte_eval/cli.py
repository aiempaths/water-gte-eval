"""Command line interface.

Usage (from the repo root):
    python -m water_gte_eval make-example-data [--out examples/data]
    python -m water_gte_eval run configs/example_mock.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from .baseline_eval import run_baseline_eval
from .devdata import generate_example_data, generate_example_processed, generate_example_raw
from .engine_run import run_engine_eval
from .ingest import run_ingest
from .inject_eval import run_inject_demo
from .prepare import run_prepare
from .report import build_report
from .runner import run_experiment


def _print_run_summary(out_dir: Path) -> None:
    metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    w = metrics["window"]
    e = metrics["event"]
    b = metrics["bootstrap"]["metrics"]
    print(f"run dir: {out_dir}")
    print(
        "window : "
        f"n={w['n_windows']} precision={w['precision']} recall={w['recall']} f1={w['f1']}"
    )
    print(
        "event  : "
        f"n={e['n_events']} precision={e['precision']} recall={e['recall']} f1={e['f1']}"
    )
    ci = metrics["bootstrap"]["ci"]
    print(
        f"boot   : unit={metrics['bootstrap']['unit']} n={metrics['bootstrap']['n_resamples']} "
        f"ci={ci} event-f1 mean={b['f1']['mean']} "
        f"[{b['f1']['ci_low']}, {b['f1']['ci_high']}]"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="water-gte-eval",
        description="Reproducible evaluation harness for a Geometric Trust Engine on water telemetry",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="run the experiment defined by one YAML config")
    r.add_argument("config", help="path to the YAML config")
    m = sub.add_parser(
        "make-example-data",
        help="generate the synthetic example dataset (NOT real plant data)",
    )
    m.add_argument("--out", default="examples/data", help="output directory (default: examples/data)")
    m2 = sub.add_parser(
        "make-example-raw",
        help="generate synthetic RAW csv files for trying the ingest pipeline (NOT real plant data)",
    )
    m2.add_argument("--out", default="examples/raw", help="output directory (default: examples/raw)")
    ing = sub.add_parser(
        "ingest",
        help="run an ingest config: raw files -> data/processed parquet + manifest.json + quality report",
    )
    ing.add_argument("config", help="path to an ingest YAML config (configs/ingest/*.yaml)")
    prep = sub.add_parser(
        "prepare",
        help="run a prepare config: processed parquet -> labeled windows -> staged split + leakage check",
    )
    prep.add_argument("config", help="path to a prepare YAML config (configs/prepare/*.yaml)")
    bl = sub.add_parser(
        "baselines",
        help="run baseline evaluation on prepared windows: runs/<config_hash>/",
    )
    bl.add_argument("config", help="path to a baselines YAML config (configs/baselines/*.yaml)")
    sw = sub.add_parser("sim-wntr", help="simulate clean hydraulic telemetry with WNTR (bundled Net3)")
    sw.add_argument("config", help="path to a sim YAML config (configs/sim/wntr_example.yaml)")
    st = sub.add_parser("sim-toy", help="simulate clean telemetry with the clearly-labeled TOY plant model")
    st.add_argument("config", help="path to a sim YAML config (configs/sim/toy_example.yaml)")
    ij = sub.add_parser("inject-demo",
                        help="inject faults into clean telemetry: example plots + severity sweep curve -> reports/")
    ij.add_argument("config", help="path to an inject YAML config (configs/inject/example.yaml)")
    ev = sub.add_parser(
        "eval",
        help="full engine+baseline eval on prepared windows -> runs/<config_hash>/",
    )
    ev.add_argument("config", help="path to a run YAML config (configs/run/*.yaml)")
    rp = sub.add_parser(
        "report",
        help="build reports/<run_hash>/report.md + PNGs from a run dir's results.parquet",
    )
    rp.add_argument("run_dir", help="run dir containing results.parquet (runs/<config_hash>/)")
    rp.add_argument("--config", default=None, help="optional report YAML (defaults built-in; see configs/report/default.yaml)")
    m3 = sub.add_parser(
        "make-example-processed",
        help="generate a synthetic processed-style parquet for trying `prepare` (NOT real plant data)",
    )
    m3.add_argument("--out", default="examples/processed_demo", help="output directory")
    args = parser.parse_args(argv)

    if args.command == "run":
        out_dir = run_experiment(args.config)
        _print_run_summary(out_dir)
        return 0
    if args.command == "make-example-data":
        paths = generate_example_data(args.out)
        print("synthetic example data written (NOT real plant measurements):")
        for k, v in paths.items():
            print(f"  {k}: {v}")
        return 0
    if args.command == "make-example-raw":
        paths = generate_example_raw(args.out)
        print("synthetic raw files written (NOT real plant data):")
        for k, v in paths.items():
            print(f"  {k}: {v}")
        return 0
    if args.command == "make-example-processed":
        paths = generate_example_processed(args.out)
        print("synthetic processed-style parquet written (NOT real plant data):")
        for k, v in paths.items():
            print(f"  {k}: {v}")
        return 0
    if args.command == "ingest":
        run_ingest(args.config)
        return 0
    if args.command == "prepare":
        run_prepare(args.config)
        return 0
    if args.command == "baselines":
        run_baseline_eval(args.config)
        return 0
    if args.command == "sim-wntr":
        from gte_eval.sim import run_wntr_sim
        run_wntr_sim(yaml.safe_load(open(args.config)))
        return 0
    if args.command == "sim-toy":
        from gte_eval.sim import run_toy_sim
        run_toy_sim(yaml.safe_load(open(args.config)))
        return 0
    if args.command == "inject-demo":
        run_inject_demo(args.config)
        return 0
    if args.command == "eval":
        run_engine_eval(args.config)
        return 0
    if args.command == "report":
        build_report(args.run_dir, args.config)
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
