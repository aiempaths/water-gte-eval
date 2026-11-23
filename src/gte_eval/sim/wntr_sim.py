"""WNTR hydraulic simulation on the bundled EPANET Net3 network.

Network used (documented): WNTR's bundled example file
  <wntr package>/library/networks/Net3.inp   (NET3_PATH below)
Net3 is the classic EPANET example network: 92 junctions, 3 tanks, 2 pumps,
119 links. EPANET/WNTR model distribution hydraulics (and simple quality
transport); they do NOT model in-plant coagulation or filtration — see
toy_plant.py for a clearly labeled toy model of those.

Each scenario = one seed -> one random 24-step hourly demand pattern assigned
to every junction demand, one EPANET run. Output is a tidy DataFrame
(DatetimeIndex from a fixed epoch, one column per tag): tank levels, selected
junction pressures, pump flows/status, selected pipe flows, total demand, and
a simple first-order chlorine decay trace C(t) = C0 * exp(-k * t_hours)
(labeled as such; NOT a full water-quality model).
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

import wntr

NET3_PATH = os.path.join(os.path.dirname(wntr.__file__), "library", "networks", "Net3.inp")
EPOCH = pd.Timestamp("2024-01-01T00:00:00")


def resolve_network(spec: str) -> str:
    """'wntr:Net3' (default, bundled) or a path to a .inp file."""
    if spec == "wntr:Net3" or spec is None:
        if not os.path.isfile(NET3_PATH):
            raise FileNotFoundError(f"bundled Net3 not found at {NET3_PATH}")
        return NET3_PATH
    if not os.path.isfile(spec):
        raise FileNotFoundError(f"network file not found: {spec}")
    return spec


def demand_pattern(rng: np.random.Generator, n_steps: int = 24) -> list[float]:
    """Random daily demand pattern: low night, morning/evening peaks with random
    phase/amplitude, normalized to mean 1."""
    hours = np.arange(n_steps)
    morning = np.exp(-((hours - rng.uniform(6, 9)) ** 2) / 2.0)
    evening = np.exp(-((hours - rng.uniform(17, 20)) ** 2) / 2.0)
    base = rng.uniform(0.6, 0.8)
    amp_m = rng.uniform(0.3, 0.7)
    amp_e = rng.uniform(0.3, 0.7)
    pat = base + amp_m * morning + amp_e * evening
    return list(pat / pat.mean())


def simulate_scenario(
    seed: int,
    network: str = "wntr:Net3",
    duration_h: float = 24.0,
    report_s: int = 900,
    base_demand_scale: float = 1.0,
    chlorine_c0: float = 1.0,
    chlorine_k_per_h: float = 0.05,
    n_pressure_nodes: int = 3,
    n_pipe_links: int = 5,
) -> pd.DataFrame:
    """Run one EPANET scenario; return a tidy telemetry DataFrame. Deterministic
    for a fixed seed."""
    inp = resolve_network(network)
    rng = np.random.default_rng(seed)
    wn = wntr.network.WaterNetworkModel(inp)
    steps = max(1, int(round(duration_h * 3600 / report_s)))
    pat = wn.add_pattern(f"scen_{seed}", demand_pattern(rng, steps)) or wn.get_pattern(f"scen_{seed}")
    pat_name = pat.name if pat is not None else f"scen_{seed}"
    for j in wn.junction_name_list:
        for d in wn.get_node(j).demand_timeseries_list:
            d._pattern = pat_name
            d.base_value = d.base_value * base_demand_scale
    wn.options.time.duration = int(duration_h * 3600)
    wn.options.time.report_timestep = int(report_s)
    wn.options.time.hydraulic_timestep = int(report_s)
    res = wntr.sim.EpanetSimulator(wn).run_sim()

    n = len(res.node["pressure"])
    idx = pd.DatetimeIndex([EPOCH + pd.Timedelta(seconds=int(t)) for t in res.node["pressure"].index])
    cols: dict[str, np.ndarray] = {}
    for t in wn.tank_name_list:
        tank = wn.get_node(t)
        level = res.node["head"][t].to_numpy() - float(tank.elevation)
        cols[f"tank_{t}_level"] = level
    for j in wn.junction_name_list[:n_pressure_nodes]:
        cols[f"node_{j}_pressure"] = res.node["pressure"][j].to_numpy()
    for p in wn.pump_name_list:
        cols[f"pump_{p}_flow"] = res.link["flowrate"][p].to_numpy()
        cols[f"pump_{p}_status"] = (res.link["status"][p].to_numpy() > 0).astype(float)
    pipes = [l for l in wn.pipe_name_list][:n_pipe_links]
    for p in pipes:
        cols[f"pipe_{p}_flow"] = res.link["flowrate"][p].to_numpy()
    cols["total_demand"] = res.node["demand"].sum(axis=1).to_numpy()
    # simple first-order chlorine decay trace at the monitoring point (NOT a WQ model)
    t_hours = np.arange(n) * report_s / 3600.0
    cols["chlorine"] = chlorine_c0 * np.exp(-chlorine_k_per_h * t_hours)
    return pd.DataFrame(cols, index=idx)


def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_wntr_sim(cfg: dict) -> Path:
    """cfg: {name, network, scenarios, seeds (or seed_base), duration_h, report_s,
    base_demand_scale, chlorine:{c0,k_per_h}, output:{dir}}. Writes one parquet
    per scenario + manifest.json. Deterministic."""
    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    network = cfg.get("network", "wntr:Net3")
    inp = resolve_network(network)
    n_scen = int(cfg.get("scenarios", 4))
    seeds = cfg.get("seeds") or [int(cfg.get("seed_base", 1)) + i for i in range(n_scen)]
    duration_h = float(cfg.get("duration_h", 24.0))
    report_s = int(cfg.get("report_s", 900))
    chlorine = cfg.get("chlorine", {})
    entries = {}
    for i, seed in enumerate(seeds):
        df = simulate_scenario(
            seed=int(seed), network=network, duration_h=duration_h, report_s=report_s,
            base_demand_scale=float(cfg.get("base_demand_scale", 1.0)),
            chlorine_c0=float(chlorine.get("c0", 1.0)),
            chlorine_k_per_h=float(chlorine.get("k_per_h", 0.05)),
        )
        p = out_dir / f"scenario_{i:02d}_seed{int(seed)}.parquet"
        df.to_parquet(p)
        entries[p.name] = {
            "seed": int(seed),
            "sha256": _file_sha256(str(p)),
            "n_rows": int(len(df)),
            "time_min": df.index.min().isoformat(),
            "time_max": df.index.max().isoformat(),
            "tags": list(df.columns),
        }
        print(f"wntr scenario seed={seed}: {len(df)} rows, {len(df.columns)} tags -> {p}")
    manifest = {
        "name": cfg.get("name", "wntr-sim"),
        "network": network,
        "network_file": inp,
        "network_sha256": _file_sha256(inp),
        "network_note": "WNTR bundled EPANET Net3 (hydraulics only; no coagulation/filtration). "
                        "Chlorine column is a simple first-order decay trace, not a WQ model.",
        "duration_h": duration_h,
        "report_s": report_s,
        "scenarios": entries,
    }
    mpath = out_dir / "manifest.json"
    mpath.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"manifest: {mpath}")
    return mpath
