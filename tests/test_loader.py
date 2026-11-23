import pandas as pd
import pytest

from water_gte_eval.data.loader import DataError, file_sha256, load_table, require_columns


def test_missing_file_error_names_path_and_formats(tmp_path):
    missing = tmp_path / "nope.parquet"
    with pytest.raises(DataError) as exc:
        load_table(missing)
    msg = str(exc.value)
    assert str(missing) in msg
    assert ".parquet" in msg and ".csv" in msg


def test_parquet_and_csv_roundtrip(tmp_path):
    df = pd.DataFrame({"t": pd.date_range("2024-01-01", periods=3, freq="5min"), "a": [1.0, 2.0, 3.0]})
    pq = tmp_path / "x.parquet"
    csv = tmp_path / "x.csv"
    df.to_parquet(pq, index=False)
    df.to_csv(csv, index=False)
    a = load_table(pq)
    b = load_table(csv)
    assert list(a.columns) == ["t", "a"] and len(a) == 3
    assert list(b.columns) == ["t", "a"] and len(b) == 3


def test_unsupported_format_rejected(tmp_path):
    p = tmp_path / "x.json"
    p.write_text("{}")
    with pytest.raises(DataError, match="Unsupported format"):
        load_table(p)


def test_require_columns_lists_available(tmp_path):
    df = pd.DataFrame({"real_ts": [1], "sensor_a": [2]})
    with pytest.raises(DataError) as exc:
        require_columns(df, ["timestamp", "sensor_a"], "Dataset file foo.parquet")
    msg = str(exc.value)
    assert "timestamp" in msg
    assert "real_ts" in msg and "sensor_a" in msg
    require_columns(df, ["real_ts", "sensor_a"], "ok")  # no error


def test_file_sha256_stable(tmp_path):
    p = tmp_path / "f.bin"
    p.write_bytes(b"abc")
    assert file_sha256(p) == file_sha256(p)
