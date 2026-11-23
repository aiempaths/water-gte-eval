"""Turn an `engine:` config section into an EngineAdapter. The only supported
types today are "mock" and "http"; register the real GTE adapter here."""

from __future__ import annotations

from .base import EngineAdapter, EngineError
from .http import HttpEngine
from .mock import MockEngine

_REGISTRY = {"mock": MockEngine, "http": HttpEngine}


def make_engine(engine_cfg: dict) -> EngineAdapter:
    etype = engine_cfg.get("type")
    params = dict(engine_cfg.get("params") or {})
    cls = _REGISTRY.get(etype)
    if cls is None:
        raise EngineError(
            f"Unknown engine type '{etype}'. Registered types: {sorted(_REGISTRY)}. "
            "To plug in the real GTE, implement gte_eval.engine_adapter.EngineAdapter "
            "and register it here; the harness only ever talks to EngineAdapter."
        )
    unknown = sorted(set(params) - set(cls.PARAMS))
    if unknown:
        raise EngineError(f"Unknown {etype} engine param(s) {unknown}; allowed: {sorted(cls.PARAMS)}")
    return cls(**params)
