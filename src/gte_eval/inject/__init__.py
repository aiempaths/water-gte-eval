"""Fault/disturbance injection into clean multi-sensor telemetry.

Each injection applies one fault type over [start, end) with a severity
parameter and records a ground-truth interval (event_id, start, end, label,
fault, severity, tags) that can be fed to the labeling stage
(water_gte_eval.data.labeling.label_windows) so injected labels match the
injected intervals exactly.

Fault types: stuck_sensor, additive_drift, constant_bias, spikes,
rising_noise, replay, delayed_dropped, actuator_fault (faults, default label
"fault") and demand_surge, tank_refill (legitimate disturbances, default
label "process_change" — NOT fault).
"""

from .faults import FAULTS, apply_fault
from .scenario import Injection, inject_scenario, severity_sweep

__all__ = ["FAULTS", "apply_fault", "Injection", "inject_scenario", "severity_sweep"]
