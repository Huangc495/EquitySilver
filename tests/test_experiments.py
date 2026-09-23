"""Experiment-grid and reporting tests (CLAUDE.md Phase 6)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from acidity_lstm.config import load_config
from acidity_lstm.evaluate import fit_line, pearson_r
from acidity_lstm.experiments import (
    PAPER_TABLE1,
    VARIANTS,
    override,
    prepare_sample_sets,
    run_variant,
)
from acidity_lstm.preprocess import clean_acidity, clean_weather
from acidity_lstm.splits import SPLIT_NAMES


@pytest.fixture(scope="module")
def data():
    cfg = load_config()
    weather, _ = clean_weather(cfg)
    acidity, _ = clean_acidity(cfg)
    return cfg, acidity, weather


def fast(cfg, **sections):
    """Config with a short training budget, for tests."""
    base = {"train": {"max_epochs": 4, "patience": 2, "n_repeats": 2}}
    base.update(sections)
    return override(cfg, base)


# --- Config overrides -----------------------------------------------------

@pytest.mark.integration
def test_override_does_not_mutate_the_original(data):
    cfg, _, _ = data
    before = cfg["split"]["method"]
    other = override(cfg, {"split": {"method": "blocked"}})
    assert other["split"]["method"] == "blocked"
    assert cfg["split"]["method"] == before


@pytest.mark.integration
def test_every_variant_is_a_valid_override(data):
    cfg, _, _ = data
    for name, updates in VARIANTS.items():
        c = override(cfg, updates)
        assert c["split"]["method"] in ("random", "blocked"), name
        assert c["scaling"]["fit_on"] in ("all", "train"), name


@pytest.mark.integration
def test_paper_variant_uses_the_unmodified_config(data):
    cfg, _, _ = data
    c = override(cfg, VARIANTS["paper"])
    assert c["split"]["method"] == "random"
    assert c["scaling"]["fit_on"] == "all"
    assert c["windows"]["common_sample_set"] is False


# --- Sample-set preparation ----------------------------------------------

@pytest.mark.integration
def test_prepare_sample_sets_assigns_every_split(data):
    cfg, acidity, weather = data
    sets = prepare_sample_sets(cfg, acidity, weather, seed=42)
    assert set(sets) == {("BD", "A"), ("BD", "B"), ("C7", "A"), ("C7", "B")}
    for key, s in sets.items():
        assert s.meta["split"].isin(SPLIT_NAMES).all(), key
        assert set(s.meta["split"]) == set(SPLIT_NAMES), key


@pytest.mark.integration
def test_blocked_variant_never_produces_an_empty_fold(data):
    """The Q-03 failure mode: empty years must not become a whole fold."""
    cfg, acidity, weather = data
    c = override(cfg, VARIANTS["blocked"])
    for seed in (0, 1, 42, 123):
        sets = prepare_sample_sets(c, acidity, weather, seed=seed)
        for key, s in sets.items():
            counts = s.meta["split"].value_counts()
            for name in SPLIT_NAMES:
                assert counts.get(name, 0) > 0, f"{key} seed={seed}: {name} is empty"


@pytest.mark.integration
def test_blocked_variant_keeps_years_intact(data):
    cfg, acidity, weather = data
    c = override(cfg, VARIANTS["blocked"])
    sets = prepare_sample_sets(c, acidity, weather, seed=42)
    for key, s in sets.items():
        per_year = s.meta.groupby(s.meta["date"].dt.year)["split"].nunique()
        assert (per_year == 1).all(), f"{key}: a year was split across folds"


@pytest.mark.integration
def test_common_variant_equalises_the_sample_sets(data):
    cfg, acidity, weather = data
    c = override(cfg, VARIANTS["common"])
    sets = prepare_sample_sets(c, acidity, weather, seed=42)
    for station in ("BD", "C7"):
        a, b = sets[(station, "A")], sets[(station, "B")]
        assert len(a) == len(b)
        assert set(a.meta["date"]) == set(b.meta["date"])


# --- Grid execution -------------------------------------------------------

@pytest.mark.integration
def test_run_variant_covers_the_whole_grid(data):
    cfg, acidity, weather = data
    results = run_variant(fast(cfg), acidity, weather, "paper",
                          mlflow=None, hidden_sizes=[5])
    assert len(results) == 4                      # 2 stations x 2 types x 1 size
    keys = {(r.station, r.window_type, r.hidden_size) for r in results}
    assert keys == {("BD", "A", 5), ("BD", "B", 5), ("C7", "A", 5), ("C7", "B", 5)}


@pytest.mark.integration
def test_scenario_row_is_complete_and_finite(data):
    cfg, acidity, weather = data
    results = run_variant(fast(cfg), acidity, weather, "paper",
                          mlflow=None, hidden_sizes=[5])
    for s in results:
        row = s.row()
        assert row["n"] == row["n_train"] + row["n_val"] + row["n_test"]
        assert np.isfinite(row["best_mse"])
        assert np.isfinite(row["rmse_mgL"]) and row["rmse_mgL"] > 0
        assert row["paper_mse"] == PAPER_TABLE1[(s.station, s.window_type)][5]
        assert row["diff"] == pytest.approx(row["best_mse"] - row["paper_mse"])


@pytest.mark.integration
def test_best_mse_is_the_minimum_over_repeats(data):
    cfg, acidity, weather = data
    results = run_variant(fast(cfg), acidity, weather, "paper",
                          mlflow=None, hidden_sizes=[5])
    for s in results:
        assert s.summary["best_all_mse"] == min(r.selection_mse for r in s.results)
        assert s.best.selection_mse == s.summary["best_all_mse"]


@pytest.mark.integration
def test_scenario_carries_metadata_for_figures(data):
    cfg, acidity, weather = data
    results = run_variant(fast(cfg), acidity, weather, "paper",
                          mlflow=None, hidden_sizes=[5])
    for s in results:
        assert s.meta is not None
        assert len(s.meta) == s.n_samples == len(s.best.evaluation.y_true_norm)
        assert {"date", "split", "acidity_mgL"} <= set(s.meta.columns)


@pytest.mark.integration
def test_run_variant_is_reproducible(data):
    cfg, acidity, weather = data
    c = fast(cfg)
    a = run_variant(c, acidity, weather, "paper", mlflow=None, hidden_sizes=[5])
    b = run_variant(c, acidity, weather, "paper", mlflow=None, hidden_sizes=[5])
    for x, y in zip(a, b):
        assert x.summary["best_all_mse"] == pytest.approx(y.summary["best_all_mse"], rel=1e-9)


# --- Figure helpers -------------------------------------------------------

def test_fit_line_recovers_a_known_slope():
    x = np.linspace(-2, 2, 50)
    a, b = fit_line(x, 0.5 * x + 1.25)
    assert a == pytest.approx(0.5)
    assert b == pytest.approx(1.25)


def test_fit_line_on_constant_x_is_nan():
    a, b = fit_line(np.ones(10), np.arange(10, dtype=float))
    assert np.isnan(a) and np.isnan(b)


@pytest.mark.integration
def test_figures_are_written(tmp_path, data):
    from acidity_lstm.evaluate import (
        scatter_measured_vs_calculated,
        timeseries_measured_vs_calculated,
    )

    cfg, acidity, weather = data
    results = run_variant(fast(cfg), acidity, weather, "paper",
                          mlflow=None, hidden_sizes=[5])
    bd_b = next(s for s in results if (s.station, s.window_type) == ("BD", "B"))

    p1 = scatter_measured_vs_calculated(
        {"BD": bd_b.best.evaluation}, tmp_path / "scatter.png", title="t"
    )
    p2 = timeseries_measured_vs_calculated(
        bd_b.meta, bd_b.best.evaluation, tmp_path / "series.png", title="t"
    )
    for p in (p1, p2):
        assert Path(p).exists() and Path(p).stat().st_size > 5000


@pytest.mark.integration
def test_scatter_r_matches_the_reported_metric(data):
    """The R drawn on Fig. 6 must be the same number the table reports."""
    cfg, acidity, weather = data
    results = run_variant(fast(cfg), acidity, weather, "paper",
                          mlflow=None, hidden_sizes=[5])
    s = results[0]
    ev = s.best.evaluation
    assert pearson_r(ev.y_true_norm, ev.y_pred_norm) == pytest.approx(
        ev.all.r, rel=1e-9
    )
