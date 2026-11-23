"""End-to-end baseline driver: prepare output -> runs/<hash>/ with shared metrics."""

import json

import pandas as pd
import pytest
import yaml

from water_gte_eval.baseline_eval import BaselineEvalConfigError, run_baseline_eval
from water_gte_eval.devdata import generate_example_processed
from water_gte_eval.prepare import run_prepare


@pytest.fixture
def prepared(tmp_path):
    processed = generate_example_processed(tmp_path / "processed_demo")["processed"]
    prep_cfg = {
        "name": "demo",
        "source": {"path": str(processed), "label_column": "attack",
                   "label_map": {"0": "normal", "1": "attack", "2": "fault"},
                   "default_label": "normal"},
        "window": {"length": 24, "stride": 6},
        "labeling": {"min_overlap_frac": 0.5, "lag": "2h"},
        "split": {"val_frac": 0.15, "test_frac": 0.15},
        "output": {"dir": str(tmp_path / "prep_out")},
    }
    p = tmp_path / "prepare.yaml"
    p.write_text(yaml.safe_dump(prep_cfg))
    run_prepare(p)
    return tmp_path / "prep_out" / "demo__example_windows.parquet"


def _write_cfg(tmp_path, windows_path, **over):
    cfg = {
        "name": "bl-demo",
        "seed": 42,
        "windows_path": str(windows_path),
        "labels": {"positive": "attack"},
        "stages": {"fit": "train", "threshold": "val", "eval": "test"},
        "baselines": ["zscore", "iforest", "autoencoder", "knn_raw", "conformal", "simple_knn_trust"],
        "params": {"autoencoder": {"hidden": 4, "max_iter": 150}},
        "threshold": {"mode": "val_best_f1"},
        "event_metrics": {"min_detected_windows": 1},
        "bootstrap": {"unit": "events", "n_resamples": 200, "ci": 0.95, "seed": 42},
        "output": {"runs_dir": str(tmp_path / "runs")},
    }
    cfg.update(over)
    p = tmp_path / "baselines.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def test_driver_end_to_end(prepared, tmp_path, capsys):
    out = run_baseline_eval(_write_cfg(tmp_path, prepared))
    assert (out / "baselines.json").is_file()
    assert (out / "baseline_scores.parquet").is_file()
    assert (out / "baseline_scores_wide.parquet").is_file()
    assert (out / "config.yaml").is_file()

    results = json.loads((out / "baselines.json").read_text())["results"]
    assert sorted(results) == ["autoencoder", "conformal", "iforest", "knn_raw",
                               "simple_knn_trust", "zscore"]
    for name, r in results.items():
        assert "threshold" in r and "val" in r and "test" in r
        assert set(r["test"]) == {"roc_auc", "window", "event", "bootstrap"}
        assert r["test"]["window"]["n_windows"] > 0
        assert r["test"]["bootstrap"]["unit"] == "events"
    # synthetic attacks are blatant -> distance/summary baselines separate well
    assert results["simple_knn_trust"]["test"]["roc_auc"] > 0.8
    assert results["zscore"]["test"]["roc_auc"] > 0.8

    wide = pd.read_parquet(out / "baseline_scores_wide.parquet")
    for name in results:
        assert f"score__{name}" in wide.columns and f"flag__{name}" in wide.columns
    assert set(wide["stage"].unique()) <= {"val", "test"}

    printed = capsys.readouterr().out
    assert "simple_knn_trust" in printed and "AUC=" in printed


def test_driver_deterministic(prepared, tmp_path):
    cfg = _write_cfg(tmp_path, prepared)
    out1 = run_baseline_eval(cfg)
    b1 = (out1 / "baselines.json").read_bytes()
    s1 = (out1 / "baseline_scores_wide.parquet").read_bytes()
    out2 = run_baseline_eval(cfg)
    assert out1 == out2
    assert (out2 / "baselines.json").read_bytes() == b1
    assert (out2 / "baseline_scores_wide.parquet").read_bytes() == s1


def test_driver_missing_windows_file(tmp_path):
    with pytest.raises(BaselineEvalConfigError, match="windows file not found"):
        run_baseline_eval(_write_cfg(tmp_path, tmp_path / "nope.parquet"))


def test_driver_unknown_baseline_rejected(prepared, tmp_path):
    cfg = _write_cfg(tmp_path, prepared, baselines=["magic"])
    with pytest.raises(BaselineEvalConfigError, match="baselines"):
        run_baseline_eval(cfg)
