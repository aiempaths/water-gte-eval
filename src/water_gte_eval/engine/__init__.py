"""The harness talks to engines ONLY through gte_eval.engine_adapter.

This module re-exports the Prompt-2 interface so the rest of the harness
(and tests) import one canonical EngineAdapter.
"""

from gte_eval.engine_adapter import (
    GENESIS,
    VERDICTS,
    EngineAdapter,
    EngineError,
    EngineResult,
    HistoryWindow,
    Window as AdapterWindow,
    make_engine,
)
from gte_eval.engine_adapter.http import EngineUnavailableError, HttpEngine
from gte_eval.engine_adapter.mock import MockEngine

__all__ = [
    "GENESIS",
    "VERDICTS",
    "EngineAdapter",
    "EngineError",
    "EngineResult",
    "HistoryWindow",
    "AdapterWindow",
    "EngineUnavailableError",
    "HttpEngine",
    "MockEngine",
    "make_engine",
]
