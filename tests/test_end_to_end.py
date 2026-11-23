"""End-to-end: one YAML config -> runs/<config_hash>/ with deterministic outputs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from water_gte_eval.cli import main
from water_gte_eval.config import config_hash, load_config
from water_gte_eval.data.loader import DataError, file_sha256
from water_gte_eval.runner import run_experiment

from conftest import base_config_dict, deep_update, write_config

OUTPUT_FILES = ["config.yaml", "manifest.json", "predictions.parquet",
                "predictions.csv", "events.csv", "metrics.json"]


def test_full_run_twice_is_byte_identical(config_path, tmp_path):
    out1 = run_experiment(config_path)
    raw = load_config(config_path).raw
    assert out1 == tmp_path / "runs" / config_hash(raw)
    assert out1.is_dir()
    bytes1 = {name: (out1 / name).read_bytes() for name in OUTPUT_FILES if name.endswith((".json", ".yaml", ".csv"))}
    preds1 = pd.read_parquet(out1 / "predictions.parquet")

    out2 = run_experiment(config_path)
    assert out2 == out1
    bytes2 = {name: (out2 / name).read_bytes() for name in bytes1}
    assert bytes1 == bytes2, "re-running the same config must give identical outputs"
    pd.testing.assert_frame_equal(preds1, pd.read_parquet(out2 / "predictions.parquet"))


def test_run_contents_and_manifest(config_path, example_data, tmp_path):
    out = run_experiment(config_path)
    for name in OUTPUT_FILES:
        assert (out / name).is_file(), name
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["engine"]["receipts_verified"] is True
    assert manifest["engine"]["type"] == "mock"
    assert manifest["engine"]["receipt_chain_head"]
    assert manifest["dataset"]["sha256"] == file_sha256(example_data["telemetry"])
    assert manifest["events"]["sha256"] == file_sha256(example_data["events"])
    assert manifest["split"]["n_eval_windows"] > 0 and manifest["split"]["n_history_windows"] > 0
    assert "created_at" not in manifest and "timestamp" not in manifest
    for name, digest in manifest["files"].items():
        assert digest == file_sha256(out / name)

    metrics = json.loads((out / "metrics.json").read_text())
    w, e, b = metrics["window"], metrics["event"], metrics["bootstrap"]
    assert w["n_windows"] == manifest["split"]["n_eval_windows"]
    assert set(w["per_scenario"])  # per-scenario breakdown present
    # 3 annotated events in the eval month; drift_holdout is held out
    assert e["n_events"] == 3
    assert b["unit"] == "events" and b["n_resamples"] == 200
    for name in ("precision", "recall", "f1"):
        assert b["metrics"][name]["ci_low"] <= b["metrics"][name]["ci_high"]

    preds = pd.read_parquet(out / "predictions.parquet")
    assert "drift_holdout" not in set(preds["scenario"])
    assert set(preds["flag"].unique()) <= {True, False}
    assert preds["receipt"].str.len().eq(64).all()
    # every scored window starts at/after eval_start
    assert (preds["start"] >= pd.Timestamp("2024-06-01T00:00:00")).all()


def test_missing_dataset_file_stops_with_instructions(tmp_path, example_data):
    cfg = base_config_dict(example_data, tmp_path / "runs")
    cfg["dataset"]["path"] = str(tmp_path / "data" / "telemetry.parquet")
    p = write_config(tmp_path, cfg)
    with pytest.raises(DataError) as exc:
        run_experiment(p)
    msg = str(exc.value)
    assert "telemetry.parquet" in msg and "Place the file at this path" in msg


def test_missing_column_lists_real_columns(tmp_path, example_data):
    cfg = base_config_dict(example_data, tmp_path / "runs")
    cfg["dataset"]["columns"]["timestamp"] = "<timestamp column>"
    p = write_config(tmp_path, cfg)
    with pytest.raises(DataError) as exc:
        run_experiment(p)
    msg = str(exc.value)
    assert "<timestamp column>" in msg and "timestamp" in msg and "flow" in msg


def test_thresholds_decision_mode(tmp_path, example_data):
    cfg = deep_update(
        base_config_dict(example_data, tmp_path / "runs"),
        {"decision": {"mode": "thresholds", "max_closeness": 0.01, "min_agreement": 1.0}},
    )
    out = run_experiment(write_config(tmp_path, cfg))
    preds = pd.read_parquet(out / "predictions.parquet")
    # with an almost-zero closeness bar, everything unfamiliar is flagged
    assert preds["flag"].mean() > 0.9


def test_cli_smoke(tmp_path, example_data):
    assert main(["make-example-data", "--out", str(tmp_path / "exdata")]) == 0
    cfg = base_config_dict(example_data, tmp_path / "runs")
    assert main(["run", str(write_config(tmp_path, cfg))]) == 0
