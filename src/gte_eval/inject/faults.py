"""Fault injection primitives. Each function modifies df in place on the rows
with start <= timestamp < end. severity scales the effect (typical range
0..2). scale per tag = std of that tag over the whole clean frame."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _interval_positions(df: pd.DataFrame, start, end) -> tuple[int, int]:
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    if e <= s:
        raise ValueError(f"end {e} must be after start {s}")
    mask = np.asarray((df.index >= s) & (df.index < e))
    pos = np.flatnonzero(mask)
    if len(pos) == 0:
        raise ValueError(f"interval [{s}, {e}) contains no samples")
    return int(pos[0]), int(pos[-1]) + 1  # half-open [a, b)


def _scale(df: pd.DataFrame, tag: str) -> float:
    sd = float(df[tag].std())
    return sd if sd > 1e-12 else 1.0


def stuck_sensor(df, start, end, severity, tags, rng, params):
    a, b = _interval_positions(df, start, end)
    n = b - a
    n_stuck = max(1, int(round(n * min(max(severity, 0.0), 1.0))))
    for tag in tags:
        v = df[tag].iloc[a]
        df.loc[df.index[a : a + n_stuck], tag] = v


def additive_drift(df, start, end, severity, tags, rng, params):
    a, b = _interval_positions(df, start, end)
    ramp = np.linspace(0.0, 1.0, b - a)
    for tag in tags:
        df.loc[df.index[a:b], tag] = df[tag].iloc[a:b] + severity * _scale(df, tag) * ramp


def constant_bias(df, start, end, severity, tags, rng, params):
    a, b = _interval_positions(df, start, end)
    for tag in tags:
        df.loc[df.index[a:b], tag] = df[tag].iloc[a:b] + severity * _scale(df, tag)


def spikes(df, start, end, severity, tags, rng, params):
    a, b = _interval_positions(df, start, end)
    n = b - a
    n_spikes = max(1, int(round(severity * 20)))
    n_spikes = min(n_spikes, n)
    pos = rng.choice(n, size=n_spikes, replace=False)
    for tag in tags:
        scale = _scale(df, tag)
        seg = df[tag].iloc[a:b].to_numpy(copy=True)
        seg[pos] += severity * scale * rng.choice([-1.0, 1.0], size=n_spikes)
        df.loc[df.index[a:b], tag] = seg


def rising_noise(df, start, end, severity, tags, rng, params):
    a, b = _interval_positions(df, start, end)
    ramp = np.linspace(0.0, 1.0, b - a)
    for tag in tags:
        scale = _scale(df, tag)
        noise = rng.normal(0.0, 1.0, b - a) * scale * (0.05 + severity * ramp)
        df.loc[df.index[a:b], tag] = df[tag].iloc[a:b] + noise


def replay(df, start, end, severity, tags, rng, params):
    """Replace [start, end) with data from an earlier window of the same length."""
    a, b = _interval_positions(df, start, end)
    n = b - a
    offset = int(params.get("replay_offset_samples", 2 * n))
    src_a = a - offset
    src_b = src_a + n
    if src_a < 0:  # clamp to the start of the frame
        src_a, src_b = 0, n
    for tag in tags:
        df.loc[df.index[a:b], tag] = df[tag].iloc[src_a:src_b].to_numpy()


def delayed_dropped(df, start, end, severity, tags, rng, params):
    """Delay each tag by shift samples and drop a severity-controlled fraction."""
    a, b = _interval_positions(df, start, end)
    shift = max(1, int(round(severity * 4)))
    drop_frac = min(0.9, 0.5 * max(severity, 0.0))
    drop_mask = rng.random(b - a) < drop_frac
    for tag in tags:
        seg = df[tag].iloc[a:b].to_numpy(copy=True)
        delayed = np.concatenate([np.full(shift, np.nan), seg[:-shift]]) if shift < len(seg) else np.full(len(seg), np.nan)
        delayed[drop_mask] = np.nan
        df.loc[df.index[a:b], tag] = delayed


def actuator_fault(df, start, end, severity, tags, rng, params):
    """Pump status inconsistent with flow. tags ignored; pump_tag/flow_tag from params.
    mode pump_off_flow_high (default): pump off while flow stays high.
    mode pump_on_flow_low: pump on while flow drops by severity."""
    pump_tag = params.get("pump_tag", "pump_status")
    flow_tag = params.get("flow_tag", "flow")
    mode = params.get("mode", "pump_off_flow_high")
    for c in (pump_tag, flow_tag):
        if c not in df.columns:
            raise ValueError(f"actuator_fault needs columns {pump_tag!r} and {flow_tag!r}; have {list(df.columns)}")
    a, b = _interval_positions(df, start, end)
    if mode == "pump_off_flow_high":
        df.loc[df.index[a:b], pump_tag] = 0.0
    elif mode == "pump_on_flow_low":
        df.loc[df.index[a:b], flow_tag] = df[flow_tag].iloc[a:b] * max(0.0, 1.0 - severity)
    else:
        raise ValueError(f"unknown actuator_fault mode {mode!r}")


def demand_surge(df, start, end, severity, tags, rng, params):
    """Legitimate demand surge -> label process_change, NOT fault."""
    a, b = _interval_positions(df, start, end)
    for tag in tags:
        df.loc[df.index[a:b], tag] = df[tag].iloc[a:b] * (1.0 + severity)


def tank_refill(df, start, end, severity, tags, rng, params):
    """Legitimate tank refill -> label process_change, NOT fault."""
    a, b = _interval_positions(df, start, end)
    ramp = np.linspace(0.0, 1.0, b - a)
    for tag in tags:
        df.loc[df.index[a:b], tag] = df[tag].iloc[a:b] + severity * _scale(df, tag) * ramp


FAULTS = {
    "stuck_sensor": (stuck_sensor, "fault"),
    "additive_drift": (additive_drift, "fault"),
    "constant_bias": (constant_bias, "fault"),
    "spikes": (spikes, "fault"),
    "rising_noise": (rising_noise, "fault"),
    "replay": (replay, "fault"),
    "delayed_dropped": (delayed_dropped, "fault"),
    "actuator_fault": (actuator_fault, "fault"),
    "demand_surge": (demand_surge, "process_change"),
    "tank_refill": (tank_refill, "process_change"),
}


def apply_fault(df: pd.DataFrame, fault: str, start, end, severity: float,
                tags: list[str], seed: int, params: dict | None = None) -> None:
    """Apply one fault to df in place. tags default: all columns (except the
    pump/status tag for actuator_fault, which uses params)."""
    if fault not in FAULTS:
        raise ValueError(f"unknown fault {fault!r}; available: {sorted(FAULTS)}")
    fn, _default_label = FAULTS[fault]
    rng = np.random.default_rng(seed)
    if tags is None:
        tags = [c for c in df.columns]
    for tag in tags:
        if tag not in df.columns:
            raise ValueError(f"tag {tag!r} not in df; columns: {list(df.columns)}")
    fn(df, pd.Timestamp(start), pd.Timestamp(end), float(severity), list(tags), rng, dict(params or {}))
