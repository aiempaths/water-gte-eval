"""Regenerate the synthetic example dataset (NOT real plant data).

    python examples/make_example_data.py [--out examples/data]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from water_gte_eval.devdata import generate_example_data

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="examples/data")
    args = ap.parse_args()
    paths = generate_example_data(args.out)
    print("synthetic example data written (NOT real plant measurements):")
    for k, v in paths.items():
        print(f"  {k}: {v}")
