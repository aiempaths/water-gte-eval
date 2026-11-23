"""Column mappings between the config and the real data files.

No column names are hard-coded anywhere: the YAML config maps semantic roles
(timestamp, sensors, event scenario/start/end/label) onto the real columns of
the user's files.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import ConfigError


@dataclass(frozen=True)
class TelemetryColumns:
    timestamp: str
    sensors: tuple[str, ...]

    @classmethod
    def from_config(cls, section: dict) -> "TelemetryColumns":
        try:
            return cls(timestamp=section["timestamp"], sensors=tuple(section["sensors"]))
        except (KeyError, TypeError) as exc:
            raise ConfigError(f"dataset.columns is invalid: {exc}") from exc


@dataclass(frozen=True)
class EventColumns:
    scenario: str
    start: str
    end: str
    label: str

    @classmethod
    def from_config(cls, section: dict) -> "EventColumns":
        try:
            return cls(
                scenario=section["scenario"],
                start=section["start"],
                end=section["end"],
                label=section["label"],
            )
        except (KeyError, TypeError) as exc:
            raise ConfigError(f"events.columns is invalid: {exc}") from exc
