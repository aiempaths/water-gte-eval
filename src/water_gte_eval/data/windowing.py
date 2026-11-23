"""Sample-based sliding windows.

Windows are defined by length and stride IN SAMPLES over the tidy frame
(DatetimeIndex, one column per tag). Window i covers rows [i*stride,
i*stride + length); its [start, end) time span is half-open (end = timestamp of
the first sample NOT in the window, or last sample + median step at the tail).
The same Window objects are handed to the engine and to every baseline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .columns import TelemetryColumns
from .windows import Window


def build_sample_windows(
    df: pd.DataFrame,
    tcols: TelemetryColumns,
    length: int,
    stride: int,
    tag_group: str = "default",
) -> list[Window]:
    if int(length) < 1 or int(stride) < 1:
        raise ValueError("window length and stride must be >= 1 sample")
    length, stride = int(length), int(stride)
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("df must have a DatetimeIndex (tidy frame from ingest)")
    n = len(df)
    if n < length:
        raise ValueError(f"not enough samples for one window: n={n} < length={length}")
    tags = list(tcols.sensors)
    idx = df.index
    if n >= 2:
        step = pd.Series(idx).diff().dropna().median()
    else:
        step = pd.Timedelta(0)
    windows: list[Window] = []
    i = 0
    for start in range(0, n - length + 1, stride):
        end_row = start + length
        w = df.iloc[start:end_row]
        end_exclusive = idx[end_row] if end_row < n else idx[-1] + step
        windows.append(
            Window(
                window_id=f"w{i:05d}",
                start=idx[start],
                end=end_exclusive,
                matrix=w[tags].to_numpy(dtype=np.float64),
                sensors=tuple(tags),
                label="",
                scenario="background",
                timestamps=tuple(w.index.tolist()),
                tag_group=tag_group,
            )
        )
        i += 1
    return windows
