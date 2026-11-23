"""Generic CSV loader: configurable time column, tag groups, label column,
resampling for sampling-rate differences, and missing-value handling."""

import numpy as np
import pandas as pd
import pytest

from water_gte_eval.data.loader import DataError
from water_gte_eval.ingest import load_generic_csv


def _swat_like(path, freq="1min", periods=120, with_gaps=True):
    ts = pd.date_range("2024-03-01", periods=periods, freq=freq)
    if with_gaps:
        ts = ts.delete([10, 11, 12])
    df = pd.DataFrame({
        "Timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
        "FIT101": 10 + np.random.default_rng(0).normal(0, 0.1, len(ts)),
        "AIT201": 5 + np.random.default_rng(1).normal(0, 0.1, len(ts)),
        "MV201": (np.arange(len(ts)) % 2).astype(float),  # flatline-ish binary
        "Attack": (np.arange(len(ts)) >= 100).astype(int),
        "Note": ["x"] * len(ts),  # non-numeric, not a tag
    })
    df.loc[5, "FIT101"] = np.nan  # missing value (written as an empty cell)
    df.to_csv(path, index=False)


def test_basic_load_and_tag_groups(tmp_path):
    f = tmp_path / "swat.csv"
    _swat_like(f)
    df, info = load_generic_csv({
        "path": str(f), "time_column": "Timestamp",
        "tag_groups": {"flow": ["FIT101"], "chem": ["AIT201", "MV201"]},
        "label_column": "Attack",
    })
    assert isinstance(df.index, pd.DatetimeIndex)
    assert set(df.columns) == {"FIT101", "AIT201", "MV201", "Attack", "source_file"}
    assert info["n_rows_raw"] == 117  # 3 timestamps were deleted before writing (the gap)
    assert info["n_rows_out"] == 117  # gaps are absent rows in the file, not NaN rows
    assert set(df["Attack"].unique()) == {0, 1}


def test_missing_column_error_lists_real_columns(tmp_path):
    f = tmp_path / "swat.csv"
    _swat_like(f)
    with pytest.raises(DataError) as exc:
        load_generic_csv({"path": str(f), "time_column": "<time column>"})
    assert "Timestamp" in str(exc.value) and "FIT101" in str(exc.value)


def test_resample_unifies_sampling_rates_and_missing_interpolated(tmp_path):
    f = tmp_path / "mixed.csv"
    _swat_like(f, freq="90s", periods=200)  # 90s native rate
    df, info = load_generic_csv({
        "path": str(f), "time_column": "Timestamp",
        "resample": "5min", "resample_agg": "mean",
        "label_column": "Attack", "missing": "interpolate",
    })
    diffs = pd.Series(df.index).diff().dropna().unique()
    assert list(diffs) == [pd.Timedelta("5min")]
    assert info["n_rows_out"] < info["n_rows_raw"]  # unified to 5min grid
    assert df["FIT101"].isna().sum() == 0  # the "" missing value was interpolated


def test_missing_drop_and_ffill(tmp_path):
    f = tmp_path / "swat.csv"
    _swat_like(f)
    df_drop, info = load_generic_csv({
        "path": str(f), "time_column": "Timestamp", "missing": "drop",
    })
    assert df_drop.isna().sum().sum() == 0
    df_ffill, _ = load_generic_csv({
        "path": str(f), "time_column": "Timestamp", "missing": "ffill", "missing_limit": 1,
    })
    assert df_ffill["FIT101"].isna().sum() == 0


def test_missing_file_error(tmp_path):
    with pytest.raises(DataError, match="CSV file not found"):
        load_generic_csv({"path": str(tmp_path / "nope.csv"), "time_column": "Timestamp"})
