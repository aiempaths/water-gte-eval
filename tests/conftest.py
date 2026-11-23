from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from water_gte_eval.devdata import generate_example_data

SENSORS = ["flow", "pressure", "level", "chlorine"]


@pytest.fixture
def example_data(tmp_path: Path) -> dict[str, Path]:
    return generate_example_data(tmp_path / "exdata", seed=42)


def base_config_dict(example_data: dict[str, Path], runs_dir: Path) -> dict:
    return {
        "name": "test-run",
        "seed": 42,
        "dataset": {
            "path": str(example_data["telemetry"]),
            "columns": {"timestamp": "timestamp", "sensors": list(SENSORS)},
        },
        "events": {
            "path": str(example_data["events"]),
            "columns": {"scenario": "scenario", "start": "start", "end": "end", "label": "label"},
        },
        "window": {"size": "1h", "step": "15min"},
        "split": {
            "history_end": "2024-06-01T00:00:00",
            "eval_start": "2024-06-01T00:00:00",
            "eval_end": "2024-07-01T00:00:00",
            "holdout_scenarios": ["drift_holdout"],
        },
        "labels": {"positive": "anomaly", "default": "normal"},
        "engine": {"type": "mock", "params": {"k": 5, "closeness_max": 2.0, "agreement_min": 0.6}},
        "decision": {"mode": "verdict", "route_verdict": "route_to_human"},
        "event_metrics": {"min_detected_windows": 1},
        "bootstrap": {"unit": "events", "n_resamples": 200, "ci": 0.95, "seed": 7},
        "output": {"runs_dir": str(runs_dir)},
    }


def deep_update(base: dict, overrides: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def write_config(tmp_path: Path, cfg: dict) -> Path:
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(cfg, sort_keys=True), encoding="utf-8")
    return p


@pytest.fixture
def config_path(tmp_path: Path, example_data: dict[str, Path]) -> Path:
    return write_config(tmp_path, base_config_dict(example_data, tmp_path / "runs"))
