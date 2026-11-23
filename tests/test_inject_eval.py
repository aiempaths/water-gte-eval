"""inject-demo driver: plots to reports/, severity curve, labels match intervals."""

import json

import pandas as pd
import pytest
import yaml

from gte_eval.sim import simulate_toy
from water_gte_eval.inject_eval import InjectDemoConfigError, run_inject_demo


@pytest.fixture
def clean_path(tmp_path):
    df = simulate_toy(seed=42, duration_h=24, dt_s=300)  # 289 samples
    p = tmp_path / "clean.parquet"
    df.to_parquet(p)
    return p


def _write_cfg(tmp_path, clean_path, out_dir):
    cfg = {
        "name": "demo",
        "seed": 42,
        "clean_path": str(clean_path),
        "window": {"length": 12, "stride": 6},
        "labeling": {"min_overlap_frac": 0.5, "lag": "1h"},
        "injections": [
            {"fault": "stuck_sensor", "start": "2024-01-01T06:00:00", "end": "2024-01-01T12:00:00",
             "severity": 1.0, "tags": ["demand_flow"], "label": "fault"},
            {"fault": "demand_surge", "start": "2024-01-01T14:00:00", "end": "2024-01-01T18:00:00",
             "severity": 0.8, "tags": ["demand_flow"], "label": "process_change"},
        ],
        "sweep": {"fault": "additive_drift", "tags": ["tank_level"],
                  "start": "2024-01-01T12:00:00", "end": "2024-01-01T20:00:00",
                  "severities": [0.1, 0.5, 2.0], "detector": "simple_knn_trust",
                  "detector_params": {"k": 5}, "flag_quantile": 0.95,
                  "min_detected_windows": 1, "label": "fault"},
        "output": {"dir": str(out_dir)},
    }
    p = tmp_path / "inject.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def test_inject_demo_outputs(clean_path, tmp_path):
    out_dir = tmp_path / "reports"
    run_inject_demo(_write_cfg(tmp_path, clean_path, out_dir))
    pngs = sorted(out_dir.glob("injected_*.png"))
    assert len(pngs) == 2
    for p in pngs:
        assert p.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"  # valid PNG
    assert (out_dir / "severity_curve.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert (out_dir / "injected_intervals.parquet").is_file()
    assert (out_dir / "inject_demo.json").is_file()

    iv = pd.read_parquet(out_dir / "injected_intervals.parquet")
    assert list(iv["label"]) == ["fault", "process_change"]  # ground truth recorded
    assert list(iv["fault"]) == ["stuck_sensor", "demand_surge"]

    summary = json.loads((out_dir / "inject_demo.json").read_text())
    assert summary["detector"] == "simple_knn_trust"
    curve = summary["sweep"]["curve"]
    assert [c["severity"] for c in curve] == [0.1, 0.5, 2.0]
    for c in curve:
        assert 0.0 <= c["event_recall"] <= 1.0
    # strong drift at severity 2.0 should be detected
    assert curve[-1]["event_recall"] >= curve[0]["event_recall"]
    assert curve[-1]["event_recall"] == 1.0


def test_inject_demo_deterministic(clean_path, tmp_path):
    cfg = _write_cfg(tmp_path, clean_path, tmp_path / "r1")
    run_inject_demo(cfg)
    cfg2 = _write_cfg(tmp_path, clean_path, tmp_path / "r2")
    # same config content but different output dir -> curve json identical except nothing path-dependent
    run_inject_demo(cfg2)
    j1 = json.loads((tmp_path / "r1" / "severity_curve.json").read_text())
    j2 = json.loads((tmp_path / "r2" / "severity_curve.json").read_text())
    assert j1 == j2
    b1 = (tmp_path / "r1" / "injected_intervals.parquet").read_bytes()
    b2 = (tmp_path / "r2" / "injected_intervals.parquet").read_bytes()
    assert b1 == b2


def test_inject_demo_missing_clean_file(tmp_path):
    cfg = _write_cfg(tmp_path, tmp_path / "nope.parquet", tmp_path / "r")
    with pytest.raises(InjectDemoConfigError, match="clean_path not found"):
        run_inject_demo(cfg)
