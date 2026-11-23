"""End-to-end report test: eval run -> build report from results.parquet alone."""

import json

import pandas as pd
import pytest
import yaml

from water_gte_eval.devdata import generate_example_processed
from water_gte_eval.engine_run import run_engine_eval
from water_gte_eval.prepare import run_prepare
from water_gte_eval.report import ReportConfigError, build_report


@pytest.fixture
def run_dir(tmp_path):
    processed = generate_example_processed(tmp_path / "pd")["processed"]
    prep_cfg = {
        "name": "demo",
        "source": {"path": str(processed), "label_column": "attack",
                   "label_map": {"0": "normal", "1": "attack", "2": "fault"},
                   "default_label": "normal"},
        "window": {"length": 24, "stride": 6},
        "labeling": {"min_overlap_frac": 0.5, "lag": "2h"},
        "split": {"val_frac": 0.15, "test_frac": 0.15},
        "output": {"dir": str(tmp_path / "prep")},
    }
    p = tmp_path / "prep.yaml"
    p.write_text(yaml.safe_dump(prep_cfg))
    run_prepare(p)
    run_cfg = {
        "name": "t", "seed": 42,
        "windows_path": str(tmp_path / "prep" / "demo__example_windows.parquet"),
        "engine": {"type": "mock", "params": {"k": 3}},
        "modes": ["frozen", "rolling"],
        "outcome": "attack_normal",
        "positive_label": "attack",
        "baselines": ["zscore", "simple_knn_trust"],
        "threshold": {"mode": "val_best_f1"},
        "resume": False,
        "event_metrics": {"min_detected_windows": 1},
        "bootstrap": {"unit": "events", "n_resamples": 50, "ci": 0.95, "seed": 42},
        "output": {"runs_dir": str(tmp_path / "runs")},
    }
    p = tmp_path / "run.yaml"
    p.write_text(yaml.safe_dump(run_cfg))
    return run_engine_eval(p)


def _report_cfg(tmp_path, **over):
    cfg = {"bootstrap_n": 100, "ci": 0.95, "seed": 42, "budgets": [0.05, 0.1, 0.2],
           "positive_labels": ["attack", "fault"], "split": "test",
           "min_detected_windows": 1, "failures_top_k": 3, "min_events_for_claims": 5,
           "output_dir": str(tmp_path / "reports")}
    cfg.update(over)
    p = tmp_path / "report.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def test_report_builds_from_results_parquet_alone(run_dir, tmp_path):
    out = build_report(run_dir, _report_cfg(tmp_path))
    res = pd.read_parquet(run_dir / "results.parquet")
    run_hash = res["config_hash"].iloc[0]
    assert out == tmp_path / "reports" / run_hash
    report_md = (out / "report.md").read_text()
    for section in ("# Evaluation report", "Per-method metrics", "Discrimination",
                    "Failures first", "Claims supported / not supported"):
        assert section in report_md
    assert "independent event" in report_md
    for fig in ("metrics_overview.png", "discrimination.png", "failures_first.png",
                "risk_coverage_engine__frozen.png", "risk_coverage_engine__rolling.png",
                "risk_coverage_baseline__zscore.png"):
        assert (out / fig).is_file() and (out / fig).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    mj = json.loads((out / "metrics.json").read_text())
    assert set(mj["methods"]) >= {"engine__frozen", "engine__rolling",
                                 "baseline__zscore", "baseline__simple_knn_trust"}
    m = mj["methods"]["engine__frozen"]
    assert "auroc" in m and "bootstrap" in m and "false_trust_at_budget" in m
    assert "0.05" in m["false_trust_at_budget"] and "0.2" in m["false_trust_at_budget"]  # json keys are strings
    assert m["events"]["n_events"] >= 1
    assert "workload_escalations_per_hour" in m


def test_report_deterministic(run_dir, tmp_path):
    cfg = _report_cfg(tmp_path)
    out1 = build_report(run_dir, cfg)
    md1 = (out1 / "report.md").read_bytes()
    mj1 = (out1 / "metrics.json").read_bytes()
    out2 = build_report(run_dir, cfg)
    assert (out2 / "report.md").read_bytes() == md1
    assert (out2 / "metrics.json").read_bytes() == mj1


def test_report_missing_results(tmp_path):
    with pytest.raises(ReportConfigError, match="results.parquet not found"):
        build_report(tmp_path / "nope", _report_cfg(tmp_path))


def test_claims_refuse_when_too_few_events(run_dir, tmp_path):
    cfg = _report_cfg(tmp_path, min_events_for_claims=10_000)
    out = build_report(run_dir, cfg)
    md = (out / "report.md").read_text()
    assert "Not supported (refused)" in md
    assert "only" in md and "independent event" in md  # refuses method-comparison claims
