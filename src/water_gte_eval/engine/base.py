"""EngineAdapter: the ONLY interface between the harness and the GTE engine.

The real engine (plugged in later) must implement this interface. The harness
never calls the engine directly; it always goes through an EngineAdapter.

Contract:
- ``add_to_history(window, label)`` stores one past labeled window. The
  harness guarantees it only ever passes windows that end strictly before the
  window about to be scored.
- ``score(window)`` returns closeness (distance to nearest past windows,
  lower = more familiar), agreement (fraction of those nearest windows sharing
  the majority outcome, higher = more consistent), a verdict, and a
  tamper-evident receipt.
- ``verify_receipt(window, result)`` must re-derive the receipt and detect
  tampering with any bound field.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..data.windows import Window


class EngineError(Exception):
    """Raised for engine misconfiguration or receipt-verification failure."""


@dataclass(frozen=True)
class ScoreResult:
    closeness: float | None  # None when no history is available
    agreement: float | None  # None when no history is available
    verdict: str
    receipt: str


class EngineAdapter(ABC):
    engine_id: str = "abstract"

    @abstractmethod
    def reset(self) -> None:
        """Clear all history and receipt-chain state."""

    @abstractmethod
    def add_to_history(self, window: Window, label: str) -> None:
        """Store a past labeled window. Must never be given a window that
        overlaps the window currently being scored (the harness enforces this)."""

    @abstractmethod
    def score(self, window: Window) -> ScoreResult:
        """Score one window against the history stored so far."""

    @abstractmethod
    def verify_receipt(self, window: Window, result: ScoreResult) -> bool:
        """Return True iff the receipt is valid for this window and result."""

    @property
    @abstractmethod
    def receipt_chain_head(self) -> str:
        """Head of the tamper-evident receipt chain (for the run manifest)."""

    def describe(self) -> dict:
        """Engine identity + params recorded in the run manifest."""
        return {"type": self.engine_id}
