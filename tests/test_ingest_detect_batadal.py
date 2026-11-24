"""BATADAL-style loader: schema detection must not assume any file/column names."""

import numpy as np
import pandas as pd
import pytest

from water_gte_eval.data.loader import DataError
from water_gte_eval.ingest import detect_schema, load_batadal_dir, load_batadal_file


def _write_csv(path, sep=",", header=("time", "alpha", "beta", "attack"), epoch=False):
    ts = pd.date_range("2024-01-01", periods=24, freq="1h")
    if epoch:
        times = ((ts - pd.Timestamp("1970-01-01")) // pd.Timedelta("1s")).astype(str)  # epoch seconds
    else:
        times = ts.strftime("%Y-%m-%d %H:%M:%S")
    rows = [sep.join(header)]
    for i, t in enumerate(times):
        rows.append(sep.join([t, str(10 + i), str(0.5 * i), "1" if i >= 12 else "0"]))
    path.write_text("\n".join(rows) + "\n")


def test_detects_schema_without_assuming_names(tmp_path):
    f = tmp_path / "whatever.dat"
    _write_csv(f, sep=";")  # odd name, semicolon delimiter, generic header
    schema = detect_schema(f)
    assert schema.delimiter == ";"
    assert schema.has_header
    assert schema.timestamp_column == "time"
    assert schema.tag_columns == ["alpha", "beta"]
    assert schema.label_columns == ["attack"]
    assert schema.n_rows == 24


def test_detects_epoch_timestamps(tmp_path):
    f = tmp_path / "epoch.csv"
    _write_csv(f, epoch=True)
    schema = detect_schema(f)
    assert schema.timestamp_column == "time"
    assert schema.timestamp_method == "epoch_s"


def test_loader_returns_tidy_frame(tmp_path):
    f = tmp_path / "scen1.csv"
    _write_csv(f)
    df, schema = load_batadal_file(f)
    assert isinstance(df.index, pd.DatetimeIndex)
    assert list(df.columns) == ["alpha", "beta", "attack"]
    assert df["alpha"].dtype == np.float64
    assert df.index.is_monotonic_increasing
    assert set(df["attack"].unique()) == {0, 1}


def test_loader_overrides(tmp_path):
    f = tmp_path / "scen2.csv"
    _write_csv(f, header=("stamp", "x", "y", "weird_name"))
    with pytest.raises(DataError, match="not in file"):
        load_batadal_file(f, overrides={"timestamp_column": "nope"})
    df, schema = load_batadal_file(f, overrides={"label_columns": ["weird_name"]})
    assert schema.label_columns == ["weird_name"]
    assert "weird_name" in df.columns


def test_dir_loader_adds_source_file_and_missing_dir_message(tmp_path):
    d = tmp_path / "batadal"
    d.mkdir()
    _write_csv(d / "a.csv")
    _write_csv(d / "b.dat", sep=";")
    (d / "notes.md").write_text("ignore me")
    loaded = load_batadal_dir(d)
    assert sorted(loaded) == ["a.csv", "b.dat"]
    assert set(loaded["a.csv"][0]["source_file"].unique()) == {"a.csv"}

    empty = tmp_path / "nope"
    with pytest.raises(DataError) as exc:
        load_batadal_dir(empty)
    msg = str(exc.value)
    assert "Place the raw BATADAL" in msg and str(empty) in msg


def test_label_value_counts_recorded(tmp_path):
    f = tmp_path / "scen.csv"
    _write_csv(f)
    schema = detect_schema(f)
    assert schema.label_value_counts["attack"] == {"0": 12, "1": 12}
    assert "label_value_counts" in schema.to_dict()
