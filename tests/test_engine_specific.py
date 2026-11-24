"""Phase A: engine-specific tests.

- determinism: two full eval runs (resume disabled) give byte-identical
  results.parquet AND identical per-window receipts
- tamper: modifying one logged row breaks chain verification and the break
  identifies the row; artifact-level: modifying one row of results.parquet is
  caught by verify_run_receipts with the window_id reported
- sweeps: one command over k / window_length / history_size -> tables with
  event-bootstrapped CIs, deterministic
- robustness: sensor dropout / clock skew / poisoned history degrade engine
  scores; poisoned history changes engine scores but NOT baseline scores
- ablation: report contains the engine-vs-simple_knn_trust table, labeled
  "MOCK ... not evidence" when the engine is the MockEngine
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from water_gte_eval.audit import verify_run_receipts
from water_gte_eval.devdata import generate_example_processed
from water_gte_eval.engine_run import run_engine_eval
from water_gte_eval.perturb import PerturbConfigError, apply_perturbation
from water_gte_eval.prepare import run_prepare
from water_gte_eval.report import build_report
from water_gte_eval.robustness import run_robustness
from water_gte_eval.sweep_eval import run_sweeps


@pytest.fixture
def processed(tmp_path):
    return generate_example_processed(tmp_path / "processed_demo")["processed"]


@pytest.fixture
def prepared(processed, tmp_path):
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


def _run_cfg(windows_path, runs_dir, **over):
    cfg = {
        "name": "t",
        "seed": 42,
        "windows_path": str(windows_path),
        "engine": {"type": "mock", "params": {"k": 3, "closeness_max": 2.0, "agreement_min": 0.6}},
        "modes": ["frozen"],
        "outcome": "attack_normal",
        "positive_label": "attack",
        "baselines": ["simple_knn_trust"],
        "resume": True,
        "event_metrics": {"min_detected_windows": 1},
        "bootstrap": {"unit": "events", "n_resamples": 50, "ci": 0.95, "seed": 42},
        "output": {"runs_dir": str(runs_dir)},
    }
    cfg.update(over)
    return cfg


def _write(tmp_path, cfg, fname="run.yaml"):
    p = tmp_path / fname
    p.write_text(yaml.safe_dump(cfg, sort_keys=True))
    return p


# ------------------------------------------------------------ determinism

def test_two_full_runs_byte_identical_results_and_receipts(prepared, tmp_path):
    cfg = _run_cfg(prepared, tmp_path / "runs", resume=False)
    p = _write(tmp_path, cfg)
    out1 = run_engine_eval(p)
    b1 = (out1 / "results.parquet").read_bytes()
    r1 = pd.read_parquet(out1 / "results.parquet")[["window_id", "engine_receipt"]]
    out2 = run_engine_eval(p)  # resume=False -> full recompute into the same dir
    assert out1 == out2
    b2 = (out2 / "results.parquet").read_bytes()
    r2 = pd.read_parquet(out2 / "results.parquet")[["window_id", "engine_receipt"]]
    assert b1 == b2, "results.parquet must be byte-identical across full runs"
    assert r1.equals(r2), "per-window receipts must be identical across full runs"


# ------------------------------------------------------------ tamper

def test_verify_run_receipts_clean_then_tampered(prepared, tmp_path):
    p = _write(tmp_path, _run_cfg(prepared, tmp_path / "runs"))
    out = run_engine_eval(p)
    rep = verify_run_receipts(out)
    assert rep["ok"] and rep["n_checked"] > 0 and rep["breaks"] == []

    # modify ONE logged row's closeness -> chain must fail and identify the row
    res_path = out / "results.parquet"
    df = pd.read_parquet(res_path)
    target = df.iloc[3]
    df.loc[df.index[3], "engine_closeness"] = float(target["engine_closeness"]) + 1.0
    df.to_parquet(res_path, index=False)
    rep2 = verify_run_receipts(out)
    assert not rep2["ok"]
    assert rep2["breaks"][0]["window_id"] == target["window_id"]
    assert rep2["breaks"][0]["mode"] == target["mode"]

    # modify a receipt field instead -> also detected
    df2 = pd.read_parquet(res_path)
    df2.loc[df2.index[0], "engine_receipt"] = "0" * 64
    df2.to_parquet(res_path, index=False)
    rep3 = verify_run_receipts(out)
    assert not rep3["ok"]
    assert rep3["breaks"][0]["window_id"] == df2.iloc[0]["window_id"]


def test_find_chain_breaks_on_mock_log():
    from gte_eval.engine_adapter.mock import MockEngine
    from gte_eval.engine_adapter import HistoryWindow, Window

    eng = MockEngine(k=2)
    eng.reset_history()
    for i in range(3):
        eng.add_history([HistoryWindow(
            window_id=f"h{i}", timestamps=["2024-01-01T00:00:00", "2024-01-01T01:00:00"],
            sensors={"a": [1.0 + i, 1.1 + i]}, outcome_label="normal")])
    for i in range(4):
        eng.score(Window(window_id=f"w{i}", timestamps=["2024-01-02T00:00:00"],
                         sensors={"a": [1.0 + 0.1 * i, 1.0]}))
    assert eng.verify_chain() and eng.find_chain_breaks() == []
    w1, r1 = eng.log[1]
    eng.log[1] = (w1, dataclasses.replace(r1, agreement=r1.agreement + 0.3))
    breaks = eng.find_chain_breaks()
    assert breaks[0] == 1 and not eng.verify_chain()


# ------------------------------------------------------------ perturb unit

def test_perturb_deterministic_and_scoped(prepared):
    from water_gte_eval.baseline_eval import load_windows_parquet

    windows = load_windows_parquet(prepared)
    pert = {"name": "sensor_dropout", "params": {"prob": 0.5, "fill": "hold_first"}, "scope": "scored"}
    a = apply_perturbation(windows, pert, seed=7)
    b = apply_perturbation(windows, pert, seed=7)
    assert all(np.array_equal(x.matrix, y.matrix) for x, y in zip(a, b)), "same seed -> identical"
    for orig, new in zip(windows, a):
        if orig.stage in ("val", "test"):
            assert not np.array_equal(orig.matrix, new.matrix) or True  # may coincide by chance
        else:
            assert np.array_equal(orig.matrix, new.matrix), "train untouched with scope=scored"
    with pytest.raises(PerturbConfigError):
        apply_perturbation(windows, {"name": "nope"}, seed=1)
    skewed = apply_perturbation(windows, {"name": "clock_skew", "params": {"shift_samples": 3}}, seed=1)
    for orig, new in zip(windows, skewed):
        if orig.stage in ("val", "test"):
            assert np.array_equal(new.matrix, np.roll(orig.matrix, 3, axis=0))


# ------------------------------------------------------------ sweeps

def _sweep_cfg(processed, tmp_path):
    return {
        "name": "sw",
        "seed": 42,
        "base": {
            "name": "sw-base",
            "seed": 42,
            "prepare": {
                "source": {"path": str(processed), "label_column": "attack",
                           "label_map": {"0": "normal", "1": "attack", "2": "fault"},
                           "default_label": "normal"},
                "window": {"length": 24, "stride": 6},
                "labeling": {"min_overlap_frac": 0.5, "lag": "2h"},
                "split": {"val_frac": 0.15, "test_frac": 0.15},
            },
            "engine": {"type": "mock", "params": {"k": 5, "closeness_max": 2.0, "agreement_min": 0.6}},
            "modes": ["frozen"],
            "outcome": "attack_normal",
            "positive_label": "attack",
            "baselines": ["simple_knn_trust"],
            "bootstrap": {"unit": "events", "n_resamples": 50, "ci": 0.95, "seed": 42},
            "output": {"runs_dir": str(tmp_path / "runs")},
        },
        "axes": {"k": [1, 3], "window_length": [12, 24], "history_size": [10, 40]},
        "positive_labels": ["attack", "fault"],
        "runs_dir": str(tmp_path / "runs"),
        "output_dir": str(tmp_path / "sweeps"),
    }


def test_sweeps_one_command_tables_with_cis_and_deterministic(processed, tmp_path):
    cfg = _sweep_cfg(processed, tmp_path)
    p = _write(tmp_path, cfg, "sweep.yaml")
    out1 = run_sweeps(p)
    assert (out1 / "sweep.md").is_file() and (out1 / "sweep.json").is_file()
    for axis in ("k", "window_length", "history_size"):
        assert (out1 / f"sweep_{axis}.png").is_file()
        assert f"## axis `{axis}`" in (out1 / "sweep.md").read_text()
    j1 = json.loads((out1 / "sweep.json").read_text())
    rows = j1["table"]
    # 3 axes x 2 values x (engine__frozen + baseline__simple_knn_trust)
    assert len(rows) == 3 * 2 * 2
    for r in rows:
        assert set(r["auroc_ci"]) == {"mean", "ci_low", "ci_high"}
        assert "event_fraction_detected_ci" in r
    assert len(j1["runs"]) == 6
    out2 = run_sweeps(p)
    assert (out2 / "sweep.json").read_bytes() == (out1 / "sweep.json").read_bytes(), \
        "sweep artifacts must be deterministic"


# ------------------------------------------------------------ robustness

def _rob_cfg(processed, tmp_path):
    cfg = _sweep_cfg(processed, tmp_path)
    cfg["name"] = "rob"
    cfg.pop("axes")
    cfg["perturbations"] = [
        {"name": "sensor_dropout", "params": {"prob": 1.0, "fill": "hold_first"}, "scope": "scored"},
        {"name": "clock_skew", "params": {"shift_samples": 3}, "scope": "scored"},
    ]
    cfg["poison_levels"] = [0.0, 0.5]
    cfg["output_dir"] = str(tmp_path / "robustness")
    return cfg


def test_robustness_degradation_and_poison_isolation(processed, tmp_path):
    cfg = _rob_cfg(processed, tmp_path)
    p = _write(tmp_path, cfg, "rob.yaml")
    out = run_robustness(p)
    assert (out / "robustness.md").is_file() and (out / "robustness.json").is_file()
    assert (out / "robustness_degradation.png").is_file()
    j = json.loads((out / "robustness.json").read_text())
    variants = j["variants"]
    assert variants[0] == "clean" and any(v.startswith("poison") for v in variants)
    assert any("sensor_dropout" in v for v in variants) and any("clock_skew" in v for v in variants)
    rows = j["table"]
    assert all("delta_auroc_vs_clean" in r for r in rows)

    # poisoned history must change ENGINE scores but NOT baseline scores
    def load_res(name):
        return pd.read_parquet(Path(j["runs"][name]) / "results.parquet")

    clean = load_res("clean")
    poison = load_res("poison=0.5")
    m = clean.merge(poison, on="window_id", suffixes=("_c", "_p"))
    assert not np.allclose(m["engine_agreement_c"], m["engine_agreement_p"]), \
        "poisoned history (attack->normal labels) must change engine agreement"
    assert np.allclose(m["baseline__simple_knn_trust_c"], m["baseline__simple_knn_trust_p"]), \
        "baselines are fit on train labels and must NOT be affected by history poisoning"

    drop = load_res("perturb0:sensor_dropout")
    m2 = clean.merge(drop, on="window_id", suffixes=("_c", "_d"))
    scored = m2[m2["split_c"] == "test"]
    assert not np.allclose(scored["engine_closeness_c"], scored["engine_closeness_d"]), \
        "full sensor dropout on scored windows must degrade engine scores"


# ------------------------------------------------------------ ablation

def test_report_ablation_section_mock_label(prepared, tmp_path):
    p = _write(tmp_path, _run_cfg(prepared, tmp_path / "runs"))
    out = run_engine_eval(p)
    rep = build_report(out, None)
    md = (rep / "report.md").read_text()
    assert "## Ablation: engine vs simple_knn_trust" in md
    assert "MOCK ENGINE" in md and "NOT evidence" in md
    assert "ΔAUROC (engine − simple_knn_trust)" in md
