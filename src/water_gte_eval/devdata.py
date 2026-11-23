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
