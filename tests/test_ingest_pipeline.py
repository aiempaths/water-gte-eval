"""End-to-end ingest pipeline: raw dir -> parquet + manifest.json + printed report."""

import json

import pandas as pd
import pytest
import yaml

from water_gte_eval.data.loader import DataError, file_sha256
from water_gte_eval.devdata import generate_example_raw
from water_gte_eval.ingest import run_ingest


@pytest.fixture
def raw_dir(tmp_path):
    generate_example_raw(tmp_path / "raw")
    return tmp_path / "raw"


def _write_cfg(tmp_path, raw_dir, out_dir, **overrides):
    cfg = {
        "name": "demo",
        "source": {"type": "batadal_dir", "path": str(raw_dir)},
        "tags_yaml": None,
        "expected_step": None,
        "flatline_min_run": 6,
        "require_units": False,
        "output": {"dir": str(out_dir)},
    }
    for k, v in overrides.items():
        cfg[k] = v
    p = tmp_path / "ingest.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def test_pipeline_writes_parquet_manifest_and_prints_report(raw_dir, tmp_path, capsys):
    out_dir = tmp_path / "processed"
    manifest_path = run_ingest(_write_cfg(tmp_path, raw_dir, out_dir))
    assert manifest_path == out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    run = manifest["runs"]["demo"]
    assert sorted(run["outputs"]) == ["demo__scenario_A", "demo__scenario_B"]

    out_a = run["outputs"]["demo__scenario_A"]
    df_a = pd.read_parquet(out_dir / "demo__scenario_A.parquet")
    assert isinstance(df_a.index, pd.DatetimeIndex)
    assert out_a["n_rows"] == len(df_a)
    assert out_a["sha256"] == file_sha256(out_dir / "demo__scenario_A.parquet")
    assert out_a["source"]["sha256"] == file_sha256(raw_dir / "scenario_A.csv")
    assert out_a["time_min"] == df_a.index.min().isoformat()
    assert out_a["time_max"] == df_a.index.max().isoformat()
    assert out_a["detected_schema"]["timestamp_column"] == "time"
    assert out_a["detected_schema"]["label_columns"] == ["attack"]
    assert out_a["tag_columns"] == ["tank_level", "flow", "pressure", "pump_state"]

    # schema B: semicolon .dat, different names detected
    out_b = run["outputs"]["demo__scenario_B"]
    assert out_b["detected_schema"]["delimiter"] == ";"
    assert out_b["detected_schema"]["label_columns"] == ["attack_label"]

    # quality report was printed and captured in the manifest
    printed = capsys.readouterr().out
    assert "data-quality report: demo__scenario_A" in printed
    assert "gaps" in printed and "flatline" in printed and "duplicate" in printed
    qa = run["quality"]["demo__scenario_A"]
    assert qa["n_gaps"] == 1 and qa["n_duplicate_rows"] == 2
    assert qa["flatline"]["pump_state"]["n_runs_ge_min"] >= 1
    qb = run["quality"]["demo__scenario_B"]
    assert qb["missing_values"]["flow"] == 3  # blank cells stay NaN in batadal_dir mode
    assert sorted(qa["unitless_tags"]) == sorted(qa["flatline"].keys())  # no tags.yaml -> all unit-less


def test_pipeline_deterministic(raw_dir, tmp_path):
    out_dir = tmp_path / "processed"
    cfg = _write_cfg(tmp_path, raw_dir, out_dir)
    run_ingest(cfg)
    b1 = (out_dir / "manifest.json").read_bytes()
    p1 = (out_dir / "demo__scenario_A.parquet").read_bytes()
    run_ingest(cfg)
    assert (out_dir / "manifest.json").read_bytes() == b1
    assert (out_dir / "demo__scenario_A.parquet").read_bytes() == p1


def test_pipeline_missing_batadal_dir_stops_with_instructions(tmp_path):
    cfg = _write_cfg(tmp_path, tmp_path / "raw_batadal", tmp_path / "processed")
    with pytest.raises(DataError) as exc:
        run_ingest(cfg)
    assert "Place the raw BATADAL" in str(exc.value)


def test_pipeline_require_units_marks_unitless_as_errors(raw_dir, tmp_path):
    # require_units turns unit-less tags into errors in the quality report
    manifest_path = run_ingest(_write_cfg(tmp_path, raw_dir, tmp_path / "p2", require_units=True))
    manifest = json.loads(manifest_path.read_text())
    qa = manifest["runs"]["demo"]["quality"]["demo__scenario_A"]
    assert qa["ok"] is False and any("unit" in e for e in qa["errors"])


def test_pipeline_csv_source(tmp_path):
    raw = tmp_path / "swat.csv"
    ts = pd.date_range("2024-03-01", periods=48, freq="1h")
    pd.DataFrame({
        "Timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
        "FIT101": range(48), "Attack": [0] * 40 + [1] * 8,
    }).to_csv(raw, index=False)
    cfg = {
        "name": "swatdemo",
        "source": {"type": "csv", "path": str(raw), "time_column": "Timestamp",
                   "label_column": "Attack", "resample": "2h", "missing": "interpolate"},
        "output": {"dir": str(tmp_path / "proc")},
    }
    p = tmp_path / "ingest_csv.yaml"
    p.write_text(yaml.safe_dump(cfg))
    manifest_path = run_ingest(p)
    manifest = json.loads(manifest_path.read_text())
    out = manifest["runs"]["swatdemo"]["outputs"]["swatdemo"]
    assert out["label_columns"] == ["Attack"]
    df = pd.read_parquet(tmp_path / "proc" / "swatdemo.parquet")
    assert (pd.Series(df.index).diff().dropna() == pd.Timedelta("2h")).all()
    assert set(df["Attack"].unique()) <= {0, 1}
