"""TOY PLANT MODEL — clearly labeled, NOT a calibrated model and NOT WNTR.

WNTR/EPANET model distribution hydraulics only; they do not model in-plant
coagulation or filtration. This toy model covers a minimal in-plant picture:
  - tank balance:        dL/dt = (Q_in - Q_out) / A
  - chlorine contact:    CSTR  dC/dt = Q*(C_in - C)/V - k*C
  - filter headloss:     dH/dt = k_f * load + noise; backwash resets H when H > H_max
  - pump:                on/off by tank-level hysteresis
Demand is a daily sinusoid with seeded noise. Use for exercising the harness
on process-like signals; do not use for real plant claims.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

EPOCH = pd.Timestamp("2024-01-01T00:00:00")


def simulate_toy(
    seed: int = 42,
    duration_h: float = 48.0,
    dt_s: int = 60,
    tank_area_m2: float = 100.0,
    tank_max_m: float = 8.0,
    pump_on_m: float = 3.0,
    pump_off_m: float = 6.5,
    pump_capacity: float = 0.05,
    cstr_volume: float = 50.0,
    c_in: float = 1.2,
    k_chlorine_per_h: float = 0.1,
    k_headloss: float = 0.002,
    headloss_max: float = 3.0,
    headloss_clean: float = 0.3,
    base_demand: float = 0.03,
    demand_amp: float = 0.02,
) -> pd.DataFrame:
    """Simulate the toy plant; return a tidy DataFrame (DatetimeIndex, tags:
    tank_level, inflow, demand_flow, pump_status, chlorine, filter_headloss).
    Deterministic for a fixed seed."""
    rng = np.random.default_rng(seed)
    n = int(duration_h * 3600 / dt_s) + 1
    dt_h = dt_s / 3600.0
    t_hours = np.arange(n) * dt_h
    demand = base_demand + demand_amp * np.sin(2 * np.pi * (t_hours - 6) / 24.0)
    demand = np.clip(demand + rng.normal(0, 0.002, n), 0.0, None)

    level = np.empty(n)
    chlorine = np.empty(n)
    headloss = np.empty(n)
    pump = np.empty(n)
    inflow = np.empty(n)
    L, C, H = 5.0, c_in, headloss_clean
    pump_on = True
    for i in range(n):
        if L <= pump_on_m:
            pump_on = True
        elif L >= pump_off_m:
            pump_on = False
        q_in = pump_capacity if pump_on else 0.0
        q_out = demand[i]
        L = L + (q_in - q_out) / tank_area_m2 * dt_h * 3600 / 1.0  # m3/s -> m over dt
        L = float(np.clip(L, 0.0, tank_max_m))
        dC = q_out * (c_in - C) / cstr_volume - k_chlorine_per_h * C * dt_h
        C = float(np.clip(C + dC, 0.0, None))
        load = q_out * 1.0  # toy: load proportional to throughput
        H = H + (k_headloss * load * 3600 * dt_h) + rng.normal(0, 0.01) * dt_h
        if H > headloss_max:  # backwash
            H = headloss_clean
        H = float(np.clip(H, 0.0, headloss_max + 1.0))
        level[i], chlorine[i], headloss[i], pump[i], inflow[i] = L, C, H, 1.0 if pump_on else 0.0, q_in
    idx = pd.DatetimeIndex([EPOCH + pd.Timedelta(seconds=i * dt_s) for i in range(n)])
    return pd.DataFrame(
        {
            "tank_level": level,
            "inflow": inflow,
            "demand_flow": demand,
            "pump_status": pump,
            "chlorine": chlorine,
            "filter_headloss": headloss,
        },
        index=idx,
    )


def _file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_toy_sim(cfg: dict) -> Path:
    """cfg: {name, scenarios, seeds/seed_base, duration_h, dt_s, output:{dir}, params...}.
    Writes telemetry parquet(s) + manifest.json. Deterministic."""
    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    n_scen = int(cfg.get("scenarios", 3))
    seeds = cfg.get("seeds") or [int(cfg.get("seed_base", 42)) + i for i in range(n_scen)]
    params = {k: v for k, v in cfg.items()
              if k not in ("name", "scenarios", "seeds", "seed_base", "output")}
    entries = {}
    for i, seed in enumerate(seeds):
        df = simulate_toy(seed=int(seed), **params)
        p = out_dir / f"toy_scenario_{i:02d}_seed{int(seed)}.parquet"
        df.to_parquet(p)
        entries[p.name] = {
            "seed": int(seed),
            "sha256": _file_sha256(p),
            "n_rows": int(len(df)),
            "time_min": df.index.min().isoformat(),
            "time_max": df.index.max().isoformat(),
            "tags": list(df.columns),
        }
        print(f"toy scenario seed={seed}: {len(df)} rows, {len(df.columns)} tags -> {p}")
    manifest = {
        "name": cfg.get("name", "toy-sim"),
        "model": "TOY PLANT MODEL (sim/toy_plant.py) - tank balance + chlorine CSTR + "
                 "filter headloss with noise. NOT calibrated, NOT WNTR, not for real plant claims.",
        "params": params,
        "scenarios": entries,
    }
    mpath = out_dir / "manifest.json"
    mpath.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"manifest: {mpath}")
    return mpath
