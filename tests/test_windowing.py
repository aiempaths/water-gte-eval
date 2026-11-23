"""Sample-based sliding windows."""

import numpy as np
import pandas as pd
import pytest

from water_gte_eval.data import TelemetryColumns, build_sample_windows
from water_gte_eval.data.windowing import build_sample_windows as bsw


def _df(n=100, freq="5min"):
    idx = pd.date_range("2024-01-01", periods=n, freq=freq)
    return pd.DataFrame({"a": np.arange(n, dtype=float), "b": np.arange(n, dtype=float) * 2}, index=idx)


TC = TelemetryColumns(timestamp="timestamp", sensors=("a", "b"))


def test_window_counts_and_ids():
    ws = bsw(_df(100), TC, length=10, stride=5)
    assert len(ws) == (100 - 10) // 5 + 1  # 19
    assert [w.window_id for w in ws] == [f"w{i:05d}" for i in range(19)]


def test_window_spans_and_arrays():
    df = _df(30)
    ws = bsw(df, TC, length=10, stride=10)
    assert len(ws) == 3
    w0 = ws[0]
    assert w0.start == df.index[0]
    assert w0.end == df.index[10]  # exclusive end = first sample NOT in the window
    assert w0.matrix.shape == (10, 2)
    assert np.array_equal(w0.sensor_arrays()["a"], df["a"].iloc[:10].to_numpy())
    assert np.array_equal(w0.sensor_arrays()["b"], df["b"].iloc[:10].to_numpy())
    assert len(w0.timestamps) == 10
    assert w0.tag_group == "default"
    # tail window: end = last sample + median step
    assert ws[-1].end == df.index[-1] + pd.Timedelta("5min")


def test_stride_smaller_than_length_gives_overlapping_windows():
    ws = bsw(_df(30), TC, length=10, stride=3)
    assert ws[1].start == ws[0].start + pd.Timedelta("15min")
    assert ws[1].start < ws[0].end  # overlapping in time by design


def test_tag_group_propagates():
    ws = bsw(_df(20), TC, length=5, stride=5, tag_group=" hydraulics")
    assert all(w.tag_group == " hydraulics" for w in ws)


def test_invalid_params():
    with pytest.raises(ValueError):
        bsw(_df(5), TC, length=10, stride=1)  # fewer samples than length
    with pytest.raises(ValueError):
        bsw(_df(20), TC, length=0, stride=1)
    with pytest.raises(ValueError):
        bsw(pd.DataFrame({"a": [1.0]}), TC, length=1, stride=1)  # no DatetimeIndex
