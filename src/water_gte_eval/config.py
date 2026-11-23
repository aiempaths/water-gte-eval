"""YAML experiment configs: loading, validation, and the run-directory hash.

Every experiment is defined by exactly one YAML config. The config's canonical
JSON encoding is hashed to ``runs/<config_hash>/`` so re-running the same
config overwrites the same run directory with identical outputs.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import yaml


class ConfigError(Exception):
    """Raised when a YAML config is missing or has invalid fields."""


def _is_str(v) -> bool:
    return isinstance(v, str) and v != ""


def _is_str_list(v) -> bool:
    return isinstance(v, list) and len(v) > 0 and all(isinstance(x, str) and x for x in v)


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise ConfigError(msg)


def validate(raw: object) -> dict:
    """Validate a raw YAML mapping and return it. Raises ConfigError with a
    precise message naming the offending field."""
    _check(isinstance(raw, dict), "config root must be a mapping")
    _check(_is_str(raw.get("name")), "'name' must be a non-empty string")
    _check(isinstance(raw.get("seed"), int) and not isinstance(raw.get("seed"), bool),
           "'seed' must be an integer")

    ds = raw.get("dataset")
    _check(isinstance(ds, dict), "'dataset' must be a mapping")
    _check(_is_str(ds.get("path")), "dataset.path must be a non-empty string")
    _check(ds.get("format") in (None, "parquet", "csv"), "dataset.format must be 'parquet' or 'csv'")
    dcols = ds.get("columns")
    _check(isinstance(dcols, dict), "dataset.columns must be a mapping")
    _check(_is_str(dcols.get("timestamp")), "dataset.columns.timestamp must be a non-empty string")
    _check(_is_str_list(dcols.get("sensors")), "dataset.columns.sensors must be a non-empty list of strings")

    ev = raw.get("events")
    _check(isinstance(ev, dict), "'events' must be a mapping")
    _check(_is_str(ev.get("path")), "events.path must be a non-empty string")
    _check(ev.get("format") in (None, "parquet", "csv"), "events.format must be 'parquet' or 'csv'")
    ecols = ev.get("columns")
    _check(isinstance(ecols, dict), "events.columns must be a mapping")
    for k in ("scenario", "start", "end", "label"):
        _check(_is_str(ecols.get(k)), f"events.columns.{k} must be a non-empty string")

    w = raw.get("window")
    _check(isinstance(w, dict), "'window' must be a mapping")
    _check(_is_str(w.get("size")), "window.size must be a duration string, e.g. '1h'")
    _check(_is_str(w.get("step")), "window.step must be a duration string, e.g. '15min'")

    sp = raw.get("split")
    _check(isinstance(sp, dict), "'split' must be a mapping")
    for k in ("history_end", "eval_start", "eval_end"):
        _check(_is_str(sp.get(k)), f"split.{k} must be a timestamp string, e.g. '2024-06-01T00:00:00'")
    hold = sp.get("holdout_scenarios", [])
    _check(isinstance(hold, list) and all(isinstance(x, str) for x in hold),
           "split.holdout_scenarios must be a list of scenario id strings")

    lb = raw.get("labels")
    _check(isinstance(lb, dict), "'labels' must be a mapping")
    _check(_is_str(lb.get("positive")), "labels.positive must be a non-empty string")
    _check(_is_str(lb.get("default")), "labels.default must be a non-empty string")

    en = raw.get("engine")
    _check(isinstance(en, dict), "'engine' must be a mapping")
    _check(_is_str(en.get("type")), "engine.type must be a non-empty string, e.g. 'mock'")
    _check(isinstance(en.get("params", {}), dict), "engine.params must be a mapping")

    dec = raw.get("decision")
    _check(isinstance(dec, dict), "'decision' must be a mapping")
    _check(dec.get("mode") in ("verdict", "thresholds"),
           "decision.mode must be 'verdict' or 'thresholds'")
    if dec.get("mode") == "verdict":
        _check(_is_str(dec.get("route_verdict")),
               "decision.route_verdict must be a non-empty string when decision.mode='verdict'")
    else:
        _check(isinstance(dec.get("max_closeness"), (int, float)),
               "decision.max_closeness must be a number when decision.mode='thresholds'")
        _check(isinstance(dec.get("min_agreement"), (int, float)),
               "decision.min_agreement must be a number when decision.mode='thresholds'")

    em = raw.get("event_metrics")
    _check(isinstance(em, dict), "'event_metrics' must be a mapping")
    mdw = em.get("min_detected_windows", 1)
    _check(isinstance(mdw, int) and not isinstance(mdw, bool) and mdw >= 1,
           "event_metrics.min_detected_windows must be an integer >= 1")

    bs = raw.get("bootstrap")
    _check(isinstance(bs, dict), "'bootstrap' must be a mapping")
    _check(bs.get("unit") in ("events", "scenarios"),
           "bootstrap.unit must be 'events' or 'scenarios' (never windows)")
    nr = bs.get("n_resamples")
    _check(isinstance(nr, int) and not isinstance(nr, bool) and nr >= 1,
           "bootstrap.n_resamples must be an integer >= 1")
    ci = bs.get("ci")
    _check(isinstance(ci, (int, float)) and 0 < ci < 1, "bootstrap.ci must be a number in (0, 1)")
    bseed = bs.get("seed")
    _check(bseed is None or (isinstance(bseed, int) and not isinstance(bseed, bool)),
           "bootstrap.seed must be null or an integer")

    out = raw.get("output")
    _check(isinstance(out, dict), "'output' must be a mapping")
    _check(_is_str(out.get("runs_dir")), "output.runs_dir must be a non-empty string")

    return raw


@dataclass
class Config:
    """A validated experiment config."""

    raw: dict
    path: Path

    @property
    def name(self) -> str:
        return self.raw["name"]

    @property
    def seed(self) -> int:
        return int(self.raw["seed"])


def load_config(path: str | Path) -> Config:
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"Config file not found: {p}")
    try:
        raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"Config file {p} is not valid YAML: {exc}") from exc
    return Config(raw=validate(raw), path=p)


def canonical_json(obj: object) -> str:
    """Deterministic JSON encoding used for hashing and receipts."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def config_hash(raw: dict) -> str:
    """Hash of the config content; the run directory is runs/<config_hash>/."""
    return hashlib.sha256(canonical_json(raw).encode("utf-8")).hexdigest()[:16]
