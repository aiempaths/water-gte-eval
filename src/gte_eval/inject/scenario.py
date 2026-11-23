"""Scenario application and severity sweeps.

inject_scenario(df, injections, seed) applies a list of injection dicts to a
copy of the clean frame and returns (modified df, intervals DataFrame with
columns event_id, start, end, label, fault, severity, tags) — the intervals
feed the labeling stage so injected labels match the injected intervals.

severity_sweep(df, ...) applies the same fault at several severities and
returns {severity: (df, intervals)}.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import pandas as pd

from .faults import FAULTS, apply_fault


@dataclass
class Injection:
    fault: str
    start: str
    end: str
    severity: float = 1.0
    tags: list | None = None
    label: str | None = None      # None -> the fault type's default label
    event_id: str | None = None   # None -> auto injNNN
    params: dict = field(default_factory=dict)

    def to_interval_row(self, idx: int) -> dict:
        default_label = FAULTS[self.fault][1]
        return {
            "event_id": self.event_id or f"inj{idx:03d}",
            "start": pd.Timestamp(self.start),
            "end": pd.Timestamp(self.end),
            "label": self.label or default_label,
            "fault": self.fault,
            "severity": float(self.severity),
            "tags": json.dumps(self.tags) if self.tags is not None else "all",
        }


def inject_scenario(df: pd.DataFrame, injections: list, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply injections (list of dicts or Injection) to a copy of df.
    Returns (modified df, intervals DataFrame). Deterministic for a fixed seed."""
    out = df.copy()
    rows = []
    for i, inj in enumerate(injections):
        if isinstance(inj, dict):
            inj = Injection(**inj)
        if inj.fault not in FAULTS:
            raise ValueError(f"unknown fault {inj.fault!r}; available: {sorted(FAULTS)}")
        apply_fault(out, inj.fault, inj.start, inj.end, inj.severity,
                    inj.tags if inj.tags is not None else list(out.columns),
                    seed=seed + i, params=inj.params)
        rows.append(inj.to_interval_row(i))
    intervals = pd.DataFrame(rows)
    if len(intervals):
        intervals = intervals[["event_id", "start", "end", "label", "fault", "severity", "tags"]]
    else:
        intervals = pd.DataFrame(columns=["event_id", "start", "end", "label", "fault", "severity", "tags"])
    return out, intervals


def severity_sweep(df: pd.DataFrame, fault: str, start, end, severities: list,
                   tags: list | None = None, label: str | None = None,
                   seed: int = 42, params: dict | None = None) -> dict:
    """Apply the same fault at each severity on a fresh copy.
    Returns {severity: (modified df, intervals DataFrame with one row)}."""
    out = {}
    for k, sev in enumerate(severities):
        mod, intervals = inject_scenario(
            df,
            [{"fault": fault, "start": str(start), "end": str(end), "severity": float(sev),
              "tags": tags, "label": label, "event_id": f"sweep_{k:03d}", "params": params or {}}],
            seed=seed + k,
        )
        out[float(sev)] = (mod, intervals)
    return out
