"""Validator: gaps, duplicates, monotonicity, flatline, unit-less tags, ranges."""

import numpy as np
import pandas as pd
import pytest

from water_gte_eval.ingest import format_quality_report, load_tags_yaml, validate_quality
from water_gte_eval.ingest.config import IngestConfigError


def _df(hours=(0, 1, 2, 4, 5), flat_tag=None):
    ts = pd.DatetimeIndex([pd.Timestamp("2024-01-01") + pd.Timedelta(hours=int(h)) for h in hours])
    data = {
        "flow": np.arange(len(hours), dtype=float) * 10,
        "level": np.full(len(hours), 7.0) if flat_tag == "level" else np.arange(len(hours), dtype=float),
    }
    if flat_tag == "flow":
        data["flow"] = np.full(len(hours), 3.0)
    return pd.DataFrame(data, index=ts)


TAGS = {"tags": {"flow": {"unit": "m3/h", "min": 0, "max": 100}, "level": {"unit": "m"}}, "settings": {}}


def test_gap_report():
    df = _df(hours=[0, 1, 2, 4, 5])  # 2h->4h gap
    r = validate_quality(df, ["flow", "level"], tags_cfg=TAGS, expected_step=pd.Timedelta("1h"))
    assert r["n_gaps"] == 1 and r["missing_periods"] == 1
    assert r["largest_gap"] == "0 days 02:00:00"
    assert r["gap_examples"][0]["after"] == "2024-01-01T02:00:00"
    assert any("gap" in w for w in r["warnings"])


def test_duplicate_detection():
    idx = pd.DatetimeIndex([pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-01"),
                            pd.Timestamp("2024-01-02")])
    df = pd.DataFrame({"flow": [1.0, 2.0, 3.0]}, index=idx)
    r = validate_quality(df, ["flow"], tags_cfg=TAGS, expected_step=pd.Timedelta("1D"))
    assert r["n_duplicate_rows"] == 2
    assert r["duplicate_examples"] == ["2024-01-01 00:00:00"]
    assert any("duplicate" in e for e in r["errors"])
    assert r["ok"] is False


def test_non_monotonic_is_error():
    idx = pd.DatetimeIndex([pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-01")])
    df = pd.DataFrame({"flow": [1.0, 2.0]}, index=idx)
    r = validate_quality(df, ["flow"], tags_cfg=TAGS)
    assert r["monotonic"] is False
    assert any("monotonically" in e for e in r["errors"])


def test_flatline_report():
    df = _df(hours=list(range(12)), flat_tag="level")
    r = validate_quality(df, ["flow", "level"], tags_cfg=TAGS, flatline_min_run=6)
    assert r["flatline"]["level"]["longest_run"] == 12
    assert r["flatline"]["level"]["n_runs_ge_min"] == 1
    assert r["flatline"]["flow"]["n_runs_ge_min"] == 0
    assert any("flatline" in w for w in r["warnings"])


def test_unitless_tags_reported_and_require_units_errors():
    df = _df(hours=[0, 1, 2])
    r = validate_quality(df, ["flow", "level"], tags_cfg={"tags": {"flow": {"unit": "m3/h"}}})
    assert r["unitless_tags"] == ["level"]
    assert r["ok"] is True  # warning only
    r2 = validate_quality(df, ["flow", "level"],
                          tags_cfg={"tags": {"flow": {"unit": "m3/h"}}}, require_units=True)
    assert any("unit" in e for e in r2["errors"])
    assert r2["ok"] is False


def test_range_violations():
    df = _df(hours=[0, 1, 2])
    df.loc[df.index[1], "flow"] = 999.0
    r = validate_quality(df, ["flow", "level"], tags_cfg=TAGS)
    assert r["range_violations"] == {"flow": 1}


def test_missing_values_reported():
    df = _df(hours=[0, 1, 2])
    df.loc[df.index[0], "flow"] = np.nan
    r = validate_quality(df, ["flow", "level"], tags_cfg=TAGS)
    assert r["missing_values"]["flow"] == 1


def test_tags_yaml_loading_and_format(tmp_path):
    p = tmp_path / "tags.yaml"
    p.write_text("tags:\n  flow: {unit: m3/h}\n")
    cfg = load_tags_yaml(p)
    assert cfg["tags"]["flow"]["unit"] == "m3/h"
    with pytest.raises(IngestConfigError):
        load_tags_yaml(tmp_path / "missing.yaml")
    r = validate_quality(_df(hours=[0, 1]), ["flow"], tags_cfg=cfg)
    text = format_quality_report("demo", r)
    assert "data-quality report: demo" in text and "unit-less tags: []" in text or "none" in text
