import numpy as np
import pandas as pd
import pytest

from water_gte_eval.data.splits import SplitError, eligible_history, temporal_split
from water_gte_eval.data.windows import Window


def make_window(i: int, start: str, end: str, scenario: str = "s", label: str = "normal") -> Window:
    return Window(
        window_id=f"w{i:05d}",
        start=pd.Timestamp(start),
        end=pd.Timestamp(end),
        matrix=np.zeros((2, 1)),
        sensors=("a",),
        label=label,
        scenario=scenario,
    )


@pytest.fixture
def windows():
    return [
        make_window(0, "2024-01-01T00:00", "2024-01-01T01:00"),
        make_window(1, "2024-01-01T01:00", "2024-01-01T02:00", scenario="hold"),
        make_window(2, "2024-01-01T02:00", "2024-01-01T03:00"),
        make_window(3, "2024-01-02T00:00", "2024-01-02T01:00"),
    ]


def test_temporal_split_by_time_and_scenario(windows):
    split = temporal_split(
        windows,
        history_end=pd.Timestamp("2024-01-01T01:00"),
        eval_start=pd.Timestamp("2024-01-01T01:00"),
        eval_end=pd.Timestamp("2024-01-02T01:00"),
        holdout_scenarios=["hold"],
    )
    assert [w.window_id for w in split.history] == ["w00000"]
    assert [w.window_id for w in split.eval] == ["w00002", "w00003"]


def test_history_must_precede_eval(windows):
    with pytest.raises(SplitError, match="history_end"):
        temporal_split(
            windows,
            history_end=pd.Timestamp("2024-01-02T00:00"),
            eval_start=pd.Timestamp("2024-01-01T01:00"),
            eval_end=pd.Timestamp("2024-01-02T01:00"),
        )


def test_empty_sides_rejected(windows):
    with pytest.raises(SplitError, match="no history windows"):
        temporal_split(windows, pd.Timestamp("2023-12-31T00:00"),
                       pd.Timestamp("2024-01-01T01:00"), pd.Timestamp("2024-01-02T01:00"))
    with pytest.raises(SplitError, match="no eval windows"):
        temporal_split(windows, pd.Timestamp("2024-01-01T01:00"),
                       pd.Timestamp("2024-01-01T01:00"), pd.Timestamp("2024-01-01T01:30"))


def test_eligible_history_is_strictly_before_scored_window():
    pool = [
        make_window(0, "2024-01-01T00:00", "2024-01-01T01:00"),
        make_window(1, "2024-01-01T00:30", "2024-01-01T01:30"),
    ]
    scored = make_window(2, "2024-01-01T01:30", "2024-01-01T02:30")
    elig = eligible_history(pool, scored)
    # window ending exactly at the scored window's start is NOT eligible
    assert [w.window_id for w in elig] == ["w00000"]
    scored2 = make_window(3, "2024-01-01T01:31", "2024-01-01T02:31")
    assert len(eligible_history(pool, scored2)) == 2
