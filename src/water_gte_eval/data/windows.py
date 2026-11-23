"""Sliding time windows over telemetry, labeled by overlap with events.

A window is ``[start, end)`` of duration ``size``, advanced by ``step``. Its
label/scenario come from the event with the largest time overlap with the
window (ties: earliest event start, then scenario id); windows overlapping no
event get the config's default label and scenario ``background``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .columns import EventColumns, TelemetryColumns

BACKGROUND = "background"


@dataclass
class Window:
    window_id: str
    start: pd.Timestamp
    end: pd.Timestamp
    matrix: np.ndarray  # (n_samples, n_sensors) float64
    sensors: tuple[str, ...]
    label: str
    scenario: str
    event_key: str | None = None
    meta: dict = field(default_factory=dict)
    timestamps: tuple = ()  # per-sample pd.Timestamp values in the window

    def feature_vector(self) -> np.ndarray:
        """Per-sensor [mean, std] summary used by the MockEngine."""
        mean = self.matrix.mean(axis=0)
        std = self.matrix.std(axis=0)
        return np.concatenate([mean, std]).astype(np.float64)


def _label_window(
    w_start: pd.Timestamp,
    w_end: pd.Timestamp,
    ev_start_ns: np.ndarray,
    ev_end_ns: np.ndarray,
    ev_scenario: list[str],
    ev_label: list[str],
    default_label: str,
) -> tuple[str, str, str | None]:
    overlap = np.minimum(ev_end_ns, w_end.value) - np.maximum(ev_start_ns, w_start.value)
    overlap = np.clip(overlap, 0, None)
    if len(overlap) == 0 or not np.any(overlap > 0):
        return default_label, BACKGROUND, None
    best = max(
        range(len(overlap)),
        key=lambda j: (overlap[j], -ev_start_ns[j], "".join(ev_scenario[j])),
    )
    if overlap[best] <= 0:
        return default_label, BACKGROUND, None
    scenario = ev_scenario[best]
    start_iso = pd.Timestamp(ev_start_ns[best]).isoformat()
    end_iso = pd.Timestamp(ev_end_ns[best]).isoformat()
    return ev_label[best], scenario, f"{scenario}|{start_iso}|{end_iso}"


def build_windows(
    df: pd.DataFrame,
    events: pd.DataFrame,
    tcols: TelemetryColumns,
    ecols: EventColumns,
    size: pd.Timedelta,
    step: pd.Timedelta,
    range_start: pd.Timestamp,
    range_end: pd.Timestamp,
    default_label: str,
) -> list[Window]:
    """Build windows with ``start + size <= range_end`` anchored at ``range_start``.

    Windows containing no telemetry rows are skipped. ``df`` must already have
    its timestamp column parsed to datetime64.
    """
    if step <= pd.Timedelta(0) or size <= pd.Timedelta(0):
        raise ValueError("window size and step must be positive durations")
    df = df.sort_values(tcols.timestamp).reset_index(drop=True)
    ts = pd.to_datetime(df[tcols.timestamp])
    ts_ns = ts.to_numpy(dtype="datetime64[ns]").view("int64")
    sensors = list(tcols.sensors)

    ev = events.copy()
    ev_start_ns = pd.to_datetime(ev[ecols.start]).to_numpy(dtype="datetime64[ns]").view("int64")
    ev_end_ns = pd.to_datetime(ev[ecols.end]).to_numpy(dtype="datetime64[ns]").view("int64")
    ev_scenario = [str(x) for x in ev[ecols.scenario].tolist()]
    ev_label = [str(x) for x in ev[ecols.label].tolist()]

    windows: list[Window] = []
    s = pd.Timestamp(range_start)
    range_end = pd.Timestamp(range_end)
    i = 0
    while s + size <= range_end:
        e = s + size
        mask = (ts_ns >= s.value) & (ts_ns < e.value)
        if mask.any():
            matrix = df.loc[mask, sensors].to_numpy(dtype=np.float64)
            sample_ts = tuple(ts.loc[mask].tolist())
            label, scenario, ekey = _label_window(
                s, e, ev_start_ns, ev_end_ns, ev_scenario, ev_label, default_label
            )
            windows.append(
                Window(
                    window_id=f"w{i:05d}",
                    start=s,
                    end=e,
                    matrix=matrix,
                    sensors=tuple(sensors),
                    label=label,
                    scenario=scenario,
                    event_key=ekey,
                    timestamps=sample_ts,
                )
            )
        i += 1
        s = s + step
    return windows
