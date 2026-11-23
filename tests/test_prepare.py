"""Prepare pipeline end-to-end: processed parquet -> windows parquet + staged split."""

import json

import pandas as pd
import pytest
import yaml

from water_gte_eval.devdata import generate_example_processed
from water_gte_eval.prepare import PrepareConfigError, run_prepare


@pytest.fixture
def processed(tmp_path):
    return generate_example_processed(tmp_path / "processed_demo")["processed"]


def _make_cfg(src, out_dir, **over):
    cfg = {
        "name": "demo",
        "source": {"path": str(src), "label_column": "attack",
                   "label_map": {"0": "normal", "1": "attack", "2": "fault"},
                   "default_label": "normal"},
        "window": {"length": 24, "stride": 6},
        "labeling": {"min_overlap_frac": 0.5, "lag": "2h"},
        "split": {"val_frac": 0.15, "test_frac": 0.15},
        "output": {"dir": str(out_dir)},
    }
    cfg.update(over)
    return cfg


def _write_cfg(tmp_path, cfg):
    p = tmp_path / "prepare.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def test_prepare_end_to_end(processed, tmp_path, capsys):
    out_dir = tmp_path / "prep_out"
    mpath = run_prepare(_write_cfg(tmp_path, _make_cfg(processed, out_dir)))
    manifest = json.loads(mpath.read_text())
    out = manifest["outputs"]["demo__example"]
    assert out["n_windows"] > 0 and out["n_intervals"] == 4  # 3 attack runs + 1 fault run
    assert out["leakage"]["ok"] is True

    wdf = pd.read_parquet(out_dir / "demo__example_windows.parquet")
    assert set(wdf.columns) >= {"window_id", "start", "end", "stage", "label", "event_id",
                               "tag_group", "flow", "pressure", "level", "chlorine"}
    assert set(wdf["stage"].unique()) <= {"train", "val", "test", "dropped"}
    assert {"train", "val", "test"} <= set(wdf["stage"].unique())
    assert out["n_windows"] == len(wdf)
    assert out["n_windows_staged"] + out["n_guard_band_dropped"] == out["n_windows"]
    assert set(wdf["label"].unique()) <= {"normal", "attack", "fault"}
    assert wdf["event_id"].ne("").sum() > 0
    # per-tag arrays stored, correct length
    assert wdf["flow"].iloc[0].__len__() == 24
    iv = pd.read_parquet(out_dir / "demo__example_intervals.parquet")
    assert list(iv.columns) == ["event_id", "start", "end", "label"]

    # window counts per split and per label are printed
    printed = capsys.readouterr().out
    assert "split summary" in printed
    for stage in ("train", "val", "test"):
        assert f"{stage}:" in printed or f"{stage} " in printed
    assert "labels:" in printed
    # split summary in manifest matches the staged window counts
    assert sum(s["n"] for s in out["split"]["stages"].values()) == out["n_windows_staged"]


def test_prepare_deterministic(processed, tmp_path):
    out_dir = tmp_path / "prep_out"
    cfg_path = _write_cfg(tmp_path, _make_cfg(processed, out_dir))
    run_prepare(cfg_path)
    b1 = (out_dir / "prepare.json").read_bytes()
    w1 = (out_dir / "demo__example_windows.parquet").read_bytes()
    run_prepare(cfg_path)
    assert (out_dir / "prepare.json").read_bytes() == b1
    assert (out_dir / "demo__example_windows.parquet").read_bytes() == w1


def test_prepare_missing_label_column_lists_columns(processed, tmp_path):
    cfg = _make_cfg(processed, tmp_path / "o")
    cfg["source"]["label_column"] = "<attack label column>"
    with pytest.raises(PrepareConfigError) as exc:
        run_prepare(_write_cfg(tmp_path, cfg))
    assert "attack" in str(exc.value) and "flow" in str(exc.value)


def test_prepare_source_not_found(tmp_path):
    with pytest.raises(PrepareConfigError, match="source.path not found"):
        run_prepare(_write_cfg(tmp_path, _make_cfg(tmp_path / "nope.parquet", tmp_path / "o")))


def test_prepare_explicit_intervals_file(processed, tmp_path):
    iv = pd.DataFrame([{
        "event_id": "evX", "start": pd.Timestamp("2024-06-10T00:00"),
        "end": pd.Timestamp("2024-06-11T00:00"), "label": "attack",
    }])
    iv_path = tmp_path / "intervals.parquet"
    iv.to_parquet(iv_path, index=False)
    cfg = _make_cfg(processed, tmp_path / "o2")
    cfg["source"].pop("label_column")
    cfg["intervals"] = {"path": str(iv_path)}
    mpath = run_prepare(_write_cfg(tmp_path, cfg))
    manifest = json.loads(mpath.read_text())
    out = manifest["outputs"]["demo__example"]
    assert out["n_intervals"] == 1
    wdf = pd.read_parquet(tmp_path / "o2" / "demo__example_windows.parquet")
    assert set(wdf.loc[wdf["event_id"] == "evX", "label"]) == {"attack"}
