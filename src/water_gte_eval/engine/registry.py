"""Engine registry: the single place that turns config into an EngineAdapter."""

from __future__ import annotations

from .base import EngineAdapter, EngineError
from .mock import MockEngine

_REGISTRY = {"mock": MockEngine}


def make_engine(engine_cfg: dict) -> EngineAdapter:
    etype = engine_cfg.get("type")
    params = dict(engine_cfg.get("params") or {})
    if etype == "mock":
        unknown = sorted(set(params) - set(MockEngine._PARAMS))
        if unknown:
            raise EngineError(
                f"Unknown mock engine param(s) {unknown}; allowed: {sorted(MockEngine._PARAMS)}"
            )
        return MockEngine(**params)
    raise EngineError(
        f"Unknown engine type '{etype}'. Registered types: {sorted(_REGISTRY)}. "
        "To plug in the real GTE, implement water_gte_eval.engine.base.EngineAdapter "
        "and register it here; the harness only ever talks to EngineAdapter."
    )
