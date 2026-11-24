"""Engine-run acceptance tests: byte-identical re-run, no test label ever visible
in history at scoring time (spy engine), thresholds from val only, resume/cache."""

import json
import subprocess

import numpy as np
import pandas as pd
import pytest
import yaml

from water_gte_eval.devdata import generate_example_processed
from water_gte_eval.engine import MockEngine
from water_gte_eval.engine_run import RunConfigError, run_engine_eval
from water_gte_eval.prepare import run_prepare


@pytest.fixture
def prepared(tmp_path):
    processed = generate_example_processed(tmp_path / "processed_demo")["processed"]
    prep_cfg = {
        "name": "demo",
        "source": {"path": str(processed), "label_column": "attack",
                   "label_map": {"0": "normal", "1": "attack", "2": "fault"},
                   "default_label": "normal"},
        "window": {"length": 24, "stride": 6},
        "labeling": {"min_overlap_frac": 0.5, "lag": "2h"},
        "split": {"val_frac": 0.15, "test_frac": 0.15},
        "output": {"dir": str(tmp_path / "prep_out")},
    }
    p = tmp_path / "prepare.yaml"
    p.write_text(yaml.safe_dump(prep_cfg))
    run_prepare(p)
    return tmp_path / "prep_out" / "demo__example_windows.parquet"


def _write_cfg(tmp_path, windows_path, out_dir, **over):
    cfg = {
        "name": "t",
        "seed": 42,
        "windows_path": str(windows_path),
        "engine": {"type": "mock", "params": {"k": 3, "closeness_max": 2.0, "agreement_min": 0.6}},
        "modes": ["frozen", "rolling", "cold_start"],
        "cold_start": {"n_history": 10},
        "outcome": "attack_normal",
        "positive_label": "attack",
        "baselines": ["zscore", "simple_knn_trust"],
        "threshold": {"mode": "val_best_f1"},
        "resume": True,
        "event_metrics": {"min_detected_windows": 1},
        "bootstrap": {"unit": "events", "n_resamples": 50, "ci": 0.95, "seed": 42},
        "output": {"runs_dir": str(out_dir)},
    }
    cfg.update(over)
    p = tmp_path / "run.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def test_full_run_schema_and_outputs(prepared, tmp_path):
    out = run_engine_eval(_write_cfg(tmp_path, prepared, tmp_path / "runs"))
    res = pd.read_parquet(out / "results.parquet")
    need = {"config_hash", "git_commit", "mode", "split", "window_id", "start", "end",
            "event_id", "true_label", "engine_closeness", "engine_agreement",
            "engine_verdict", "engine_receipt", "engine_neighbors", "engine_flag",
            "baseline__zscore", "baseline__simple_knn_trust"}
    assert need <= set(res.columns)
    assert set(res["mode"].unique()) == {"frozen", "rolling", "cold_start"}
    assert set(res["split"].unique()) <= {"val", "test"}
    commit = subprocess.run(["git", "-C", "/workspace", "rev-parse", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    assert (res["git_commit"] == commit).all()
    assert (res["engine_receipt"].str.len() == 64).all()

    log = pd.read_parquet(out / "engine_log.parquet")
    assert "latency_ms" in log.columns and (log["latency_ms"] >= 0).all()
    for f in ("config.yaml", "metrics.json", "manifest.json", "environment.json"):
        assert (out / f).is_file()
    env = json.loads((out / "environment.json").read_text())
    assert env["git_commit"] == commit and "pandas" in env["packages"]
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["files"]["results.parquet"] == __import__("hashlib").sha256(
        (out / "results.parquet").read_bytes()).hexdigest()
    metrics = json.loads((out / "metrics.json").read_text())
    assert set(metrics["modes"]) == {"frozen", "rolling", "cold_start"}
    assert metrics["threshold_policy"].startswith("chosen on val")
    for name, bm in metrics["baselines"].items():
        assert bm["threshold_source"] == "val" and "test" in bm


def test_rerun_byte_identical_results_parquet(prepared, tmp_path):
    cfg = _write_cfg(tmp_path, prepared, tmp_path / "runs")
    out1 = run_engine_eval(cfg)
    b1 = (out1 / "results.parquet").read_bytes()
    m1 = (out1 / "metrics.json").read_bytes()
    out2 = run_engine_eval(cfg)
    assert out1 == out2
    assert (out2 / "results.parquet").read_bytes() == b1  # ACCEPTANCE: byte-identical
    assert (out2 / "metrics.json").read_bytes() == m1
    manifest = json.loads((out2 / "manifest.json").read_text())
    assert set(manifest["cache"]["hits"]) == {"engine_frozen", "engine_rolling",
                                              "engine_cold_start", "baselines"}  # resume hit


def test_thresholds_chosen_on_val_never_test(prepared, tmp_path):
    out = run_engine_eval(_write_cfg(tmp_path, prepared, tmp_path / "runs"))
    res = pd.read_parquet(out / "results.parquet")
    metrics = json.loads((out / "metrics.json").read_text())
    from water_gte_eval.baseline_eval import best_threshold
    for name in ("zscore", "simple_knn_trust"):
        thr = metrics["baselines"][name]["threshold"]
        val = res[res["split"] == "val"].drop_duplicates("window_id")
        y = [1 if l == "attack" else 0 for l in val["true_label"]]
        expected_thr, _ = best_threshold([float(s) for s in val[f"baseline__{name}"]], y)
        assert thr == expected_thr  # exactly the val-fitted threshold
        # test flags in results.parquet are score >= frozen threshold (not re-fit on test)
        test = res[(res["split"] == "test") & (res["mode"] == "frozen")]
        assert (test[f"baseline__{name}__flag"] == (test[f"baseline__{name}"] >= thr)).all()


class SpyEngine(MockEngine):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.seen = []

    def _score_impl(self, window):
        self.seen.append((window, [(h[0].window_id, h[0].timestamps[0], h[0].timestamps[-1],
                                     h[0].outcome_label) for h in self._history]))
        return super()._score_impl(window)


@pytest.mark.parametrize("outcome", ["attack_normal", "next_window_state", "fault_type"])
@pytest.mark.parametrize("mode", ["frozen", "rolling", "cold_start"])
def test_no_test_label_ever_in_history_at_scoring(prepared, tmp_path, monkeypatch, outcome, mode):
    spies = []

    def factory(engine_cfg):
        s = SpyEngine(**engine_cfg["params"])
        spies.append(s)
        return s

    monkeypatch.setattr("water_gte_eval.engine_run.make_engine", factory)
    cfg = _write_cfg(tmp_path, prepared, tmp_path / "runs", modes=[mode], outcome=outcome)
    run_engine_eval(cfg)

    wdf = pd.read_parquet(prepared)
    label_of = dict(zip(wdf["window_id"], wdf["label"]))
    start_of = dict(zip(wdf["window_id"], pd.to_datetime(wdf["start"])))
    stage_of = dict(zip(wdf["window_id"], wdf["stage"]))
    scored_ids = [w for w in wdf[wdf["stage"].isin(["val", "test"])]
                  .sort_values("start")["window_id"]]
    # next_window_state is WITHIN a stage: the next window of the last val window
    # is NOT the first test window (crossing would leak a test label into history)
    next_label = {}
    for stage in ("val", "test"):
        ids = [w for w in scored_ids if stage_of[w] == stage]
        for i, w in enumerate(ids):
            next_label[w] = label_of[ids[i + 1]] if i + 1 < len(ids) else label_of[w]

    assert len(spies) == 1
    spy = spies[0]
    assert spy.seen, "engine scored nothing"
    for window, history in spy.seen:
        w_start = pd.Timestamp(window.timestamps[0])
        for h_id, h_start_iso, h_end_iso, h_label in history:
            # 1. time invariant: history ends strictly before the scored window starts
            assert pd.Timestamp(h_end_iso) < w_start, f"history {h_id} overlaps scored {window.window_id}"
            # 2. no future window (esp. test) is in history
            assert pd.Timestamp(h_start_iso) < w_start
            if stage_of.get(h_id) == "test":
                # 3. a test window's label may only be history AFTER that test window was scored
                assert start_of[h_id] < w_start
                if outcome == "next_window_state":
                    assert h_label == next_label[h_id]
                    if stage_of[h_id] == "val":
                        # a val window's history label must never be a TEST window's
                        # label (cross-stage next_window_state leak)
                        test_labels = {label_of[w] for w in scored_ids if stage_of[w] == "test"}
                        last_val = max((w for w in scored_ids if stage_of[w] == "val"),
                                       key=lambda w: start_of[w])
                        if h_id == last_val:
                            assert h_label == label_of[h_id], \
                                "last val window must fall back to its OWN label, not the first test window's"
                        assert h_label not in test_labels or h_label == label_of[h_id]
                    nxt_stage = [w for w in scored_ids if stage_of[w] == stage_of[h_id]]
                    if h_id in nxt_stage:
                        i = nxt_stage.index(h_id)
                        nxt = nxt_stage[i + 1] if i + 1 < len(nxt_stage) else h_id
                        assert start_of[nxt] <= w_start
    # first test window scored: history contains NO test windows at all
    first_test = next(w for w, _ in spy.seen if stage_of[w.window_id] == "test")
    hist_at_first_test = next(h for w, h in spy.seen if w.window_id == first_test.window_id)
    assert all(stage_of[h[0]] != "test" for h in hist_at_first_test)
    # cold_start subsample respected
    if mode == "cold_start":
        assert len(spies[0].seen[0][1]) == 10


def test_next_window_state_never_crosses_stage_boundary(prepared, tmp_path, monkeypatch):
    """Regression: rolling + next_window_state must label the last val window with
    its OWN label when it enters history during test scoring — never with the
    first test window's label (cross-stage leak)."""
    df = pd.read_parquet(prepared)
    first_test_idx = df[df["stage"] == "test"].sort_values("start").index[0]
    df.loc[first_test_idx, "label"] = "attack"  # distinct from the (normal) val labels
    first_test = str(df.loc[first_test_idx, "window_id"])
    p = tmp_path / "w.parquet"
    df.to_parquet(p, index=False)

    spies = []

    class SpyEngine(MockEngine):
        def __init__(self, **kw):
            super().__init__(**kw)
            self.seen = []

        def _score_impl(self, window):
            self.seen.append((window, [(h[0].window_id, h[0].outcome_label) for h in self._history]))
            return super()._score_impl(window)

    def factory(engine_cfg):
        s = SpyEngine(**engine_cfg["params"])
        spies.append(s)
        return s

    monkeypatch.setattr("water_gte_eval.engine_run.make_engine", factory)
    cfg = _write_cfg(tmp_path, p, tmp_path / "runs", modes=["rolling"], outcome="next_window_state")
    run_engine_eval(cfg)

    wdf = pd.read_parquet(p)
    label_of = dict(zip(wdf["window_id"], wdf["label"]))
    stage_of = dict(zip(wdf["window_id"], wdf["stage"]))
    start_of = dict(zip(wdf["window_id"], pd.to_datetime(wdf["start"])))
    last_val = max((w for w in wdf["window_id"] if stage_of[w] == "val"), key=lambda w: start_of[w])
    assert label_of[last_val] != label_of[first_test]
    leaked = []
    for window, history in spies[0].seen:
        if stage_of[window.window_id] != "test":
            continue
        for h_id, h_label in history:
            if h_id == last_val and h_label != label_of[last_val]:
                leaked.append((window.window_id, h_id, h_label))
    assert not leaked, f"cross-stage next_window_state leaked a test label into history: {leaked}"


def test_eval_enforces_leakage_check_on_windows_file(prepared, tmp_path):
    from water_gte_eval.data import LeakageError

    df = pd.read_parquet(prepared)
    # tamper: move one val window into train -> time overlap / shared event across stages
    idx = df.index[df["stage"] == "val"][0]
    df.loc[idx, "stage"] = "train"
    bad = tmp_path / "leaky_windows.parquet"
    df.to_parquet(bad, index=False)
    with pytest.raises(LeakageError):
        run_engine_eval(_write_cfg(tmp_path, bad, tmp_path / "runs"))


def test_missing_windows_path_error(tmp_path):
    cfg = _write_cfg(tmp_path, tmp_path / "nope.parquet", tmp_path / "r")
    with pytest.raises(Exception, match="windows file not found|windows_path"):
        run_engine_eval(cfg)


def test_cache_invalidated_on_windows_change(prepared, tmp_path):
    cfg = _write_cfg(tmp_path, prepared, tmp_path / "runs")
    out = run_engine_eval(cfg)
    assert json.loads((out / "manifest.json").read_text())["cache"]["computed"]
    # change the windows file content -> cache invalid -> recompute
    df = pd.read_parquet(prepared)
    df["flow"] = df["flow"].map(lambda a: np.asarray(a) * 1.01)
    df.to_parquet(prepared)
    out2 = run_engine_eval(cfg)
    man = json.loads((out2 / "manifest.json").read_text())
    assert man["cache"]["computed"] and not man["cache"]["hits"]
