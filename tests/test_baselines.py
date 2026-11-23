"""Baseline unit tests on synthetic data where the answer is known:
normal windows ~ base 1.0, attack windows ~ base 100.0 -> every baseline must
score attacks much higher (untrust) than normals, and be deterministic."""

import time

import numpy as np
import pandas as pd
import pytest

from gte_eval.baselines import (
    REGISTRY,
    AutoencoderSummary,
    IForestSummary,
    KNNDistanceRaw,
    SimpleKNNTrust,
    SplitConformal,
    ZScoreLimits,
    make_baseline,
)
from gte_eval.baselines.base import BaselineError, NotFittedError
from gte_eval.engine_adapter import Window


def make_window(i: int, base: float, length: int = 12, seed: int = 0) -> Window:
    rng = np.random.default_rng(seed + i)
    ts0 = pd.Timestamp("2024-01-01T00:00:00")
    ts = [(ts0 + pd.Timedelta(hours=j)).isoformat() for j in range(length)]
    sensors = {
        "a": list(base + rng.normal(0, 0.01, length)),
        "b": list(2 * base + rng.normal(0, 0.01, length)),
    }
    return Window(window_id=f"w{i:04d}", timestamps=ts, sensors=sensors)


@pytest.fixture
def data():
    train = [make_window(i, 1.0, seed=1000) for i in range(60)]
    train_labels = ["normal"] * 60
    attack = [make_window(5000 + i, 100.0, seed=2000) for i in range(5)]
    normal = [make_window(6000 + i, 1.0, seed=3000) for i in range(5)]
    return train, train_labels, attack, normal


ALL = ["zscore", "iforest", "autoencoder", "knn_raw", "simple_knn_trust"]


@pytest.mark.parametrize("name", ALL)
def test_attack_scores_higher_than_normal(name, data):
    train, train_labels, attack, normal = data
    b = make_baseline({"name": name, "seed": 42, "params": {}})
    b.fit(train, train_labels)
    sa = float(np.mean(b.score_windows(attack)))
    sn = float(np.mean(b.score_windows(normal)))
    assert sa > sn, f"{name}: attack untrust {sa} not > normal {sn}"
    assert sa > 5 * sn or sa > 10, f"{name}: separation too small (attack={sa}, normal={sn})"


def test_zscore_limit_check_known_answer(data):
    train, train_labels, attack, normal = data
    b = ZScoreLimits(seed=42, limit=3.0)
    b.fit(train, train_labels)
    s_attack = b.score(attack[0])
    s_normal = b.score(normal[0])
    assert s_attack > 3.0  # beyond the limit
    assert s_normal < 3.0  # within the limit
    assert s_attack > 100  # ~99 sigma away


def test_iforest_deterministic_and_seeded(data):
    train, train_labels, attack, normal = data
    b1 = IForestSummary(seed=7, n_estimators=50)
    b2 = IForestSummary(seed=7, n_estimators=50)
    b1.fit(train, train_labels)
    b2.fit(train, train_labels)
    assert b1.score_windows(attack + normal) == b2.score_windows(attack + normal)


def test_autoencoder_reconstruction_error(data):
    train, train_labels, attack, normal = data
    b = AutoencoderSummary(seed=3, hidden=4, max_iter=200)
    b.fit(train, train_labels)
    sa = b.score(attack[0])
    sn = b.score(normal[0])
    assert sa > sn


def test_knn_raw_on_downsampled(data):
    train, train_labels, attack, normal = data
    b = KNNDistanceRaw(seed=42, k=5, downsample=8)
    b.fit(train, train_labels)
    assert b.score(attack[0]) > b.score(normal[0])


def test_conformal_needs_labels_and_scores_attack_higher():
    # train with interleaved labels so the contiguous fit half has both classes
    normals = [make_window(i, 1.0, seed=1000) for i in range(40)]
    attacks = [make_window(100 + i, 100.0, seed=2000) for i in range(20)]
    train = [w for pair in zip(normals, attacks) for w in pair] + normals[20:]
    labels = [l for pair in zip(["normal"] * 20, ["attack"] * 20) for l in pair] + ["normal"] * 20
    attack = [make_window(5000 + i, 100.0, seed=3000) for i in range(3)]
    normal = [make_window(6000 + i, 1.0, seed=4000) for i in range(3)]
    b = SplitConformal(seed=42, cal_frac=0.5, positive_label="attack")
    b.fit(train, labels)
    sa = float(np.mean(b.score_windows(attack)))
    sn = float(np.mean(b.score_windows(normal)))
    assert sa > sn
    assert 0.0 <= b.conformal_threshold(alpha=0.1) <= 1.0


def test_conformal_single_class_train_is_constant():
    train = [make_window(i, 1.0, seed=1000) for i in range(30)]
    b = SplitConformal(seed=42)
    b.fit(train, ["normal"] * 30)
    s = b.score(make_window(9999, 100.0, seed=5000))
    assert s == pytest.approx(0.0)  # P(attack) = 0 when train has no attacks


def test_simple_knn_trust_two_score_parts(data):
    train, train_labels, attack, normal = data
    b = SimpleKNNTrust(seed=42, k=5)
    b.fit(train, train_labels)
    c_norm, a_norm = b.score_parts(normal[0])
    c_att, a_att = b.score_parts(attack[0])
    assert c_att > c_norm          # closeness: attack is far from history
    assert a_norm == pytest.approx(1.0)  # all neighbors normal -> agree
    assert 0.0 <= a_att <= 1.0
    assert b.score(attack[0]) > b.score(normal[0])
    # untrust = closeness + (1 - agreement)
    assert b.score(normal[0]) == pytest.approx(c_norm + (1 - a_norm))


def test_simple_knn_trust_agreement_drops_on_mixed_history():
    # history: normals at 1.0 and attacks at 100.0; query in the middle -> mixed neighbors
    train = [make_window(i, 1.0, seed=1000) for i in range(20)] + \
            [make_window(100 + i, 100.0, seed=2000) for i in range(20)]
    labels = ["normal"] * 20 + ["attack"] * 20
    b = SimpleKNNTrust(seed=42, k=4)
    b.fit(train, labels)
    mid = make_window(9000, 50.0, seed=6000)
    _, agreement = b.score_parts(mid)
    assert agreement < 1.0  # neighbors disagree


@pytest.mark.parametrize("name", ALL + ["conformal"])
def test_determinism_same_seed_same_scores(name, data):
    train, train_labels, attack, normal = data
    if name == "conformal":
        train = train + [make_window(100 + i, 100.0, seed=2000) for i in range(10)]
        train_labels = train_labels + ["attack"] * 10
    b1 = make_baseline({"name": name, "seed": 42, "params": {}})
    b2 = make_baseline({"name": name, "seed": 42, "params": {}})
    b1.fit(train, train_labels)
    b2.fit(train, train_labels)
    assert b1.score_windows(attack + normal) == b2.score_windows(attack + normal)


def test_not_fitted_raises(data):
    b = ZScoreLimits()
    with pytest.raises(NotFittedError):
        b.score(data[2][0])


def test_registry():
    assert sorted(REGISTRY) == ["autoencoder", "conformal", "iforest", "knn_raw",
                                "simple_knn_trust", "zscore"]
    for name in REGISTRY:
        b = make_baseline({"name": name, "seed": 1, "params": {}})
        assert b.name == name
    with pytest.raises(BaselineError, match="Unknown baseline"):
        make_baseline({"name": "magic"})
    with pytest.raises(BaselineError, match="Unknown zscore param"):
        make_baseline({"name": "zscore", "params": {"bogus": 1}})


def test_perf_smoke_many_windows():
    # acceptance: BATADAL-scale windows (a few thousand) in well under a few minutes
    rng = np.random.default_rng(0)
    ts0 = pd.Timestamp("2024-01-01T00:00:00")
    n = 2400
    train = []
    for i in range(n):
        ts = [(ts0 + pd.Timedelta(minutes=5 * j)).isoformat() for j in range(12)]
        base = 100.0 if i % 50 == 0 else 1.0
        sensors = {t: list(base + rng.normal(0, 0.1, 12)) for t in ("a", "b", "c", "d")}
        train.append(Window(window_id=f"w{i:05d}", timestamps=ts, sensors=sensors))
    labels = ["attack" if i % 50 == 0 else "normal" for i in range(n)]
    query = train[:50]
    t0 = time.time()
    for name in REGISTRY:
        b = make_baseline({"name": name, "seed": 42, "params": {}})
        b.fit(train, labels)
        b.score_windows(query)
    elapsed = time.time() - t0
    assert elapsed < 90, f"all baselines took {elapsed:.1f}s on {n} windows"
