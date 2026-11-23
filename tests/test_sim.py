"""Sim tests: WNTR Net3 run (documented network) + clearly-labeled toy plant."""

import json

import numpy as np
import pandas as pd
import pytest

from gte_eval.sim import NET3_PATH, run_toy_sim, run_wntr_sim, simulate_scenario, simulate_toy


def test_net3_bundled_path_documented():
    import os
    assert NET3_PATH.endswith(os.path.join("library", "networks", "Net3.inp"))
    assert os.path.isfile(NET3_PATH)


def test_wntr_scenario_columns_and_decay():
    df = simulate_scenario(seed=1, duration_h=6, report_s=900)
    assert isinstance(df.index, pd.DatetimeIndex)
    assert len(df) == 6 * 4 + 1
    assert any(c.startswith("tank_") and c.endswith("_level") for c in df.columns)
    assert any(c.startswith("pump_") and c.endswith("_status") for c in df.columns)
    assert any(c.startswith("node_") and c.endswith("_pressure") for c in df.columns)
    assert "total_demand" in df.columns and "chlorine" in df.columns
    # simple first-order decay trace: strictly decreasing
    assert np.all(np.diff(df["chlorine"].to_numpy()) <= 0)
    assert set(df.filter(like="pump_").filter(like="status").iloc[:, 0].unique()) <= {0.0, 1.0}


def test_wntr_deterministic_and_seed_dependent():
    a = simulate_scenario(seed=5, duration_h=3, report_s=1800)
    b = simulate_scenario(seed=5, duration_h=3, report_s=1800)
    c = simulate_scenario(seed=6, duration_h=3, report_s=1800)
    pd.testing.assert_frame_equal(a, b)
    assert not a["total_demand"].equals(c["total_demand"])  # different demand pattern


def test_run_wntr_sim_writes_manifest(tmp_path):
    cfg = {"name": "t", "network": "wntr:Net3", "scenarios": 2, "seed_base": 7,
           "duration_h": 2, "report_s": 1800, "output": {"dir": str(tmp_path / "wntr")}}
    mpath = run_wntr_sim(cfg)
    m = json.loads(mpath.read_text())
    assert m["network_file"].endswith("Net3.inp") and len(m["network_sha256"]) == 64
    assert len(m["scenarios"]) == 2
    for entry in m["scenarios"].values():
        df = pd.read_parquet(tmp_path / "wntr" / list(m["scenarios"])[0])
        assert entry["n_rows"] == len(df)


def test_toy_plant_deterministic_and_physical():
    a = simulate_toy(seed=3, duration_h=12, dt_s=300)
    b = simulate_toy(seed=3, duration_h=12, dt_s=300)
    pd.testing.assert_frame_equal(a, b)
    assert set(a.columns) == {"tank_level", "inflow", "demand_flow", "pump_status",
                              "chlorine", "filter_headloss"}
    assert a["tank_level"].between(0, 8.0).all()
    assert a["chlorine"].ge(0).all()
    assert a["filter_headloss"].between(0, 4.0).all()
    assert set(a["pump_status"].unique()) <= {0.0, 1.0}
    # tank balance: level stays within bounds and reacts to pump hysteresis
    assert a["tank_level"].nunique() > 5


def test_run_toy_sim_manifest(tmp_path):
    cfg = {"name": "toy", "scenarios": 2, "seed_base": 9, "duration_h": 6, "dt_s": 600,
           "output": {"dir": str(tmp_path / "toy")}}
    mpath = run_toy_sim(cfg)
    m = json.loads(mpath.read_text())
    assert "TOY" in m["model"] and len(m["scenarios"]) == 2
