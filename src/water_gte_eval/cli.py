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

from .devdata import generate_example_data
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
    return 2


if __name__ == "__main__":
    sys.exit(main())
