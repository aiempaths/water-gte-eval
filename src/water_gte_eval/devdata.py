"""Synthetic example dataset generator.

This produces clearly-labeled SYNTHETIC data so the harness runs end-to-end
with the MockEngine before real plant data is available. It is NOT real
telemetry and must never be used for real conclusions. For real experiments,
place your files under data/ and fill in configs/water_gte.yaml.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

EXAMPLE_EVENTS = [
    ("pump_fault", "2024-06-03T02:00:00", "2024-06-03T06:00:00", "anomaly"),
    ("chlorine_spike", "2024-06-10T12:00:00", "2024-06-10T15:00:00", "anomaly"),
    ("maintenance", "2024-06-20T08:00:00", "2024-06-20T12:00:00", "normal"),
    ("drift_holdout", "2024-06-25T00:00:00", "2024-06-25T04:00:00", "anomaly"),
]


def generate_example_data(out_dir: str | Path, seed: int = 42) -> dict[str, Path]:
    """Write telemetry.parquet and events.parquet into ``out_dir``. Deterministic
    for a fixed seed."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2024-05-01T00:00:00")
    end = pd.Timestamp("2024-07-01T00:00:00")
    ts = pd.date_range(start, end, freq="5min", inclusive="left")
    n = len(ts)
    hour = ts.hour.to_numpy() + ts.minute.to_numpy() / 60.0
    flow = 120 + 15 * np.sin(2 * np.pi * (hour - 6) / 24) + rng.normal(0, 1.5, n)
    pressure = 3.2 + 0.2 * np.sin(2 * np.pi * hour / 24) + rng.normal(0, 0.05, n)
    level = 7.5 + rng.normal(0, 0.05, n)
    chlorine = 0.8 + rng.normal(0, 0.02, n)

    for scenario, s, e, _label in EXAMPLE_EVENTS:
        m = (ts >= pd.Timestamp(s)) & (ts < pd.Timestamp(e))
        if scenario == "pump_fault":
            flow[m] *= 0.6
            pressure[m] -= 0.8
        elif scenario == "chlorine_spike":
            chlorine[m] *= 3.0
        elif scenario == "maintenance":
            flow[m] *= 1.1
        elif scenario == "drift_holdout":
            level[m] += np.linspace(0.0, 0.8, int(m.sum()))

    telemetry = pd.DataFrame(
        {
            "timestamp": ts,
            "flow": flow,
            "pressure": pressure,
            "level": level,
            "chlorine": chlorine,
        }
    )
    events = pd.DataFrame(EXAMPLE_EVENTS, columns=["scenario", "start", "end", "label"])
    events["start"] = pd.to_datetime(events["start"])
    events["end"] = pd.to_datetime(events["end"])

    tpath = out / "telemetry.parquet"
    epath = out / "events.parquet"
    telemetry.to_parquet(tpath, index=False)
    events.to_parquet(epath, index=False)
    return {"telemetry": tpath, "events": epath}


def generate_example_raw(out_dir: str | Path, seed: int = 42) -> dict[str, Path]:
    """Write two synthetic RAW csv files (generic names on purpose, to exercise
    schema detection) into ``out_dir``. NOT real plant data. Deterministic."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    # scenario A: comma csv, hourly, with a 3h gap, a duplicate row, and a flatline
    ts = pd.date_range("2024-05-01T00:00:00", "2024-05-10T23:00:00", freq="1h")
    ts = ts.delete([100, 101, 102])  # 3h gap
    n = len(ts)
    level = 7.0 + rng.normal(0, 0.05, n)
    flow = 100 + rng.normal(0, 2.0, n)
    pressure = 3.0 + rng.normal(0, 0.05, n)
    pump = np.zeros(n)  # flatline
    attack = np.zeros(n)
    a = pd.DataFrame({
        "time": ts.strftime("%Y-%m-%d %H:%M:%S"),
        "tank_level": level, "flow": flow, "pressure": pressure,
        "pump_state": pump, "attack": attack,
    })
    a = pd.concat([a, a.iloc[[5]]], ignore_index=True)  # duplicate timestamp row
    p1 = out / "scenario_A.csv"
    a.to_csv(p1, index=False)

    # scenario B: semicolon csv, 30-min, with missing values and an attack window
    ts2 = pd.date_range("2024-06-01T00:00:00", "2024-06-05T23:30:00", freq="30min")
    n2 = len(ts2)
    level2 = 7.0 + rng.normal(0, 0.05, n2)
    flow2 = 100 + rng.normal(0, 2.0, n2)
    flow2[50:53] = np.nan  # missing values
    attack2 = np.zeros(n2)
    attack2[60:72] = 1  # attack window
    b = pd.DataFrame({
        "timestamp": ts2.strftime("%Y-%m-%dT%H:%M:%S"),
        "level": level2, "flow": flow2, "attack_label": attack2,
    })
    p2 = out / "scenario_B.dat"
    b.to_csv(p2, index=False, sep=";")

    return {"scenario_A": p1, "scenario_B": p2}


def generate_example_processed(out_dir: str | Path, seed: int = 42) -> dict[str, Path]:
    """Write a tidy processed-style parquet (DatetimeIndex, tag columns, attack
    label column with attack/fault runs) into ``out_dir`` for trying `prepare`.
    NOT real plant data. Deterministic."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-05-01T00:00:00", "2024-06-30T23:00:00", freq="1h")
    n = len(idx)
    hour = idx.hour.to_numpy() + idx.minute.to_numpy() / 60.0
    flow = 120 + 15 * np.sin(2 * np.pi * (hour - 6) / 24) + rng.normal(0, 1.5, n)
    pressure = 3.2 + 0.2 * np.sin(2 * np.pi * hour / 24) + rng.normal(0, 0.05, n)
    level = 7.5 + rng.normal(0, 0.05, n)
    chlorine = 0.8 + rng.normal(0, 0.02, n)
    attack = np.zeros(n, dtype=int)
    m = (idx >= pd.Timestamp("2024-06-03")) & (idx < pd.Timestamp("2024-06-05"))
    attack[m] = 1
    flow[m] *= 0.6
    pressure[m] -= 0.8
    m2 = (idx >= pd.Timestamp("2024-06-10")) & (idx < pd.Timestamp("2024-06-11"))
    attack[m2] = 1
    chlorine[m2] *= 3.0
    m3 = (idx >= pd.Timestamp("2024-06-20")) & (idx < pd.Timestamp("2024-06-21"))
    attack[m3] = 2
    level[m3] += 0.5
    m4 = (idx >= pd.Timestamp("2024-06-25")) & (idx < pd.Timestamp("2024-06-26"))
    attack[m4] = 1
    flow[m4] *= 0.7
    df = pd.DataFrame(
        {"flow": flow, "pressure": pressure, "level": level, "chlorine": chlorine, "attack": attack},
        index=idx,
    )
    df.index.name = "timestamp"
    p = out / "example.parquet"
    df.to_parquet(p)
    return {"processed": p}
