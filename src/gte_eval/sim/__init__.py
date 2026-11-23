"""Simulation of clean water-distribution telemetry.

- wntr_sim: hydraulic telemetry from WNTR running the bundled EPANET Net3
  network (see wntr_sim.NET3_PATH) with per-scenario demand patterns, plus a
  SIMPLE first-order chlorine decay trace (not a full water-quality model).
  WNTR/EPANET model hydraulics only — they do NOT model in-plant coagulation
  or filtration.
- toy_plant: a clearly labeled TOY in-plant model (tank balance, chlorine
  contact CSTR, filter headloss growth with noise) for processes WNTR does
  not cover. Not calibrated, not for real plant claims.
"""

from .wntr_sim import NET3_PATH, simulate_scenario, run_wntr_sim
from .toy_plant import simulate_toy, run_toy_sim

__all__ = ["NET3_PATH", "simulate_scenario", "run_wntr_sim", "simulate_toy", "run_toy_sim"]
