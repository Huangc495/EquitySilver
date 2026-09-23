"""FC baseline and time-tag experiment tests (CLAUDE.md Phases 7-8)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from acidity_lstm.config import load_config
from acidity_lstm.experiments import (
    TIME_TAG_MODES,
    override,
    prepare_sample_sets,
    run_fc_baseline,
    run_refined,
    scenarios_to_table,
)
from acidity_lstm.preprocess import clean_acidity, clean_weather
from acidity_lstm.scaling import fit_scaler_for


# Every test here reads the real data (DEVIATIONS.md D-34).
pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def data():
    cfg = load_config()
    weather, _ = clean_weather(cfg)
    acidity, _ = clean_acidity(cfg)
    return cfg, acidity, weather


def fast(cfg, **sections):
    base = {"train": {"max_epochs": 4, "patience": 2, "n_repeats": 2}}
    base.update(sections)
    return override(cfg, base)


# --- Phase 7 --------------------------------------------------------------

def test_fc_baseline_runs_both_models_per_station(data):
    cfg, acidity, weather = data
    sc = run_fc_baseline(fast(cfg), acidity, weather, mlflow=None)
    assert set(sc) == {("BD", "lstm"), ("BD", "fc"), ("C7", "lstm"), ("C7", "fc")}
    for (station, kind), s in sc.items():
        assert s.kind == kind
        assert s.window_type == "B", "the baseline comparison uses Type B"


def test_fc_and_lstm_share_samples_splits_and_scaler(data):
    """Required for their metrics to be comparable at all (D-18)."""
    cfg, acidity, weather = data
    sc = run_fc_baseline(fast(cfg), acidity, weather, mlflow=None)
    for station in ("BD", "C7"):
        lstm, fc = sc[(station, "lstm")], sc[(station, "fc")]
        assert lstm.n_samples == fc.n_samples
        assert lstm.split_counts == fc.split_counts
        assert lstm.scaler.y_mean == fc.scaler.y_mean
        assert lstm.scaler.y_std == fc.scaler.y_std
        assert np.array_equal(
            lstm.best.evaluation.y_true_norm, fc.best.evaluation.y_true_norm
        )


def test_fc_baseline_parameter_count(data):
    """20*10 + 10 + 10 + 1 = 221."""
    cfg, acidity, weather = data
    sc = run_fc_baseline(fast(cfg), acidity, weather, mlflow=None)
    assert sc[("BD", "fc")].best.n_parameters == 221
    assert sc[("BD", "lstm")].best.n_parameters == 571


def test_paper_mse_is_not_claimed_for_the_fc_baseline(data):
    """Table 1 covers the LSTM only."""
    cfg, acidity, weather = data
    sc = run_fc_baseline(fast(cfg), acidity, weather, mlflow=None)
    assert np.isnan(sc[("BD", "fc")].paper_mse)
    assert not np.isnan(sc[("BD", "lstm")].paper_mse)


# --- Phase 8 --------------------------------------------------------------

def test_refined_runs_all_tag_modes(data):
    cfg, acidity, weather = data
    sc = run_refined(fast(cfg), acidity, weather, mlflow=None, stations=("BD",))
    assert set(sc) == {("BD", "none"), ("BD", "per_step"), ("BD", "constant")}
    assert sc[("BD", "none")].best.model.n_features == 2
    for mode in TIME_TAG_MODES:
        assert sc[("BD", mode)].best.model.n_features == 3
        assert sc[("BD", mode)].time_tag == mode


def test_tag_does_not_change_the_sample_set_or_output_scaler(data):
    """Why tagged and untagged MSEs are comparable (D-18)."""
    cfg, acidity, weather = data
    sc = run_refined(fast(cfg), acidity, weather, mlflow=None, stations=("BD",))
    plain = sc[("BD", "none")]
    for mode in TIME_TAG_MODES:
        tagged = sc[("BD", mode)]
        assert tagged.n_samples == plain.n_samples
        assert tagged.split_counts == plain.split_counts
        assert tagged.scaler.y_mean == pytest.approx(plain.scaler.y_mean)
        assert tagged.scaler.y_std == pytest.approx(plain.scaler.y_std)


def test_paper_mse_is_not_claimed_for_the_refined_model(data):
    cfg, acidity, weather = data
    sc = run_refined(fast(cfg), acidity, weather, mlflow=None, stations=("BD",))
    assert np.isnan(sc[("BD", "per_step")].paper_mse)


def test_tag_encodings_differ_by_at_most_one_window(data):
    """per_step and constant differ by at most 63 days for Type B."""
    cfg, acidity, weather = data
    sets = {}
    for mode in TIME_TAG_MODES:
        c = override(cfg, {"windows": {"time_tag": mode}})
        sets[mode] = prepare_sample_sets(c, acidity, weather, seed=42,
                                         with_time_tag=True)[("BD", "B")]

    diff = np.abs(sets["per_step"].X[:, :, 2] - sets["constant"].X[:, :, 2])
    assert diff.max() == pytest.approx(63.0), "9 weeks x 7 days"

    # That difference is negligible once the tag is z-scored.
    scaler = fit_scaler_for(sets["per_step"], cfg)
    assert diff.max() / scaler.x_std[2] < 0.05


def test_within_window_tag_spread_is_tiny_versus_between_sample(data):
    """The measured reason the two tag modes give the same model."""
    cfg, acidity, weather = data
    c = override(cfg, {"windows": {"time_tag": "per_step"}})
    s = prepare_sample_sets(c, acidity, weather, seed=42,
                            with_time_tag=True)[("BD", "B")]
    tag = fit_scaler_for(s, c).transform_x(s.X)[:, :, 2]

    within = tag.std(axis=1).mean()
    between = tag.mean(axis=1).std()
    assert within / between < 0.05


def test_constant_tag_is_constant_within_each_window(data):
    cfg, acidity, weather = data
    c = override(cfg, {"windows": {"time_tag": "constant"}})
    s = prepare_sample_sets(c, acidity, weather, seed=42,
                            with_time_tag=True)[("BD", "B")]
    assert np.allclose(s.X[:, :, 2].std(axis=1), 0.0)


def test_scenarios_to_table_round_trips(data):
    cfg, acidity, weather = data
    sc = run_refined(fast(cfg), acidity, weather, mlflow=None, stations=("BD",))
    t = scenarios_to_table(sc)
    assert len(t) == 3
    assert set(t["time_tag"]) == {"", "per_step", "constant"}
    assert (t["n"] == t["n_train"] + t["n_val"] + t["n_test"]).all()


def test_blocked_variant_of_the_refined_model_runs(data):
    """The Phase 8 robustness check must not hit an empty fold."""
    cfg, acidity, weather = data
    sc = run_refined(fast(cfg), acidity, weather, mlflow=None,
                     stations=("BD",), variant="blocked")
    for key, s in sc.items():
        assert all(v > 0 for v in s.split_counts.values()), key
        assert np.isfinite(s.best.evaluation.all.mse)
