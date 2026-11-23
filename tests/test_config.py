from pathlib import Path

import pytest

from water_gte_eval.config import ConfigError, config_hash, load_config, validate

REPO = Path(__file__).resolve().parents[1]


def test_example_config_loads_and_validates():
    cfg = load_config(REPO / "configs" / "example_mock.yaml")
    assert cfg.name == "example-mock"
    assert cfg.seed == 42
    assert cfg.raw["engine"]["type"] == "mock"


def test_config_hash_is_deterministic_and_sensitive():
    cfg = load_config(REPO / "configs" / "example_mock.yaml")
    h1 = config_hash(cfg.raw)
    h2 = config_hash(load_config(REPO / "configs" / "example_mock.yaml").raw)
    assert h1 == h2 and len(h1) == 16 and all(c in "0123456789abcdef" for c in h1)
    changed = dict(cfg.raw)
    changed["seed"] = 43
    assert config_hash(changed) != h1


def test_missing_seed_rejected():
    raw = {"name": "x"}
    with pytest.raises(ConfigError, match="seed"):
        validate(raw)


def test_bad_decision_mode_rejected():
    raw = {
        "name": "x", "seed": 1,
        "dataset": {"path": "d.parquet", "columns": {"timestamp": "t", "sensors": ["a"]}},
        "events": {"path": "e.parquet", "columns": {"scenario": "s", "start": "a", "end": "b", "label": "l"}},
        "window": {"size": "1h", "step": "15min"},
        "split": {"history_end": "2024-01-01", "eval_start": "2024-01-02", "eval_end": "2024-02-01"},
        "labels": {"positive": "p", "default": "n"},
        "engine": {"type": "mock"},
        "decision": {"mode": "random"},
        "event_metrics": {"min_detected_windows": 1},
        "bootstrap": {"unit": "events", "n_resamples": 10, "ci": 0.95},
        "output": {"runs_dir": "runs"},
    }
    with pytest.raises(ConfigError, match="decision.mode"):
        validate(raw)


def test_bootstrap_unit_must_not_be_windows():
    raw = {
        "name": "x", "seed": 1,
        "dataset": {"path": "d.parquet", "columns": {"timestamp": "t", "sensors": ["a"]}},
        "events": {"path": "e.parquet", "columns": {"scenario": "s", "start": "a", "end": "b", "label": "l"}},
        "window": {"size": "1h", "step": "15min"},
        "split": {"history_end": "2024-01-01", "eval_start": "2024-01-02", "eval_end": "2024-02-01"},
        "labels": {"positive": "p", "default": "n"},
        "engine": {"type": "mock"},
        "decision": {"mode": "verdict", "route_verdict": "route_to_human"},
        "event_metrics": {"min_detected_windows": 1},
        "bootstrap": {"unit": "windows", "n_resamples": 10, "ci": 0.95},
        "output": {"runs_dir": "runs"},
    }
    with pytest.raises(ConfigError, match="bootstrap.unit"):
        validate(raw)


def test_missing_config_file():
    with pytest.raises(ConfigError, match="not found"):
        load_config("/nonexistent/config.yaml")
