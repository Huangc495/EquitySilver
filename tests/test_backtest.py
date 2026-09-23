"""Backtest, promotion gate, refit and the 5-seed ensemble (MLOPS.md M3; D9, D10)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from acidity_lstm.backtest import (
    Recipe,
    backtest,
    candidate_recipe,
    fold_labels,
    gate,
    recipe_from_tags,
    refit,
    year_folds,
)
from acidity_lstm.config import load_config
from acidity_lstm.experiments import override
from acidity_lstm.models import predict
from acidity_lstm.serving import build_network, predict_members


@pytest.fixture
def cfg():
    return load_config()


def fast(cfg, **gate):
    """Small settings so a synthetic backtest runs in seconds."""
    return override(cfg, {"train": {"max_epochs": 8, "patience": 3},
                          "gate": {"n_folds": 3, "n_seeds": 2, "val_years": 1, **gate}})


def synthetic_data(years=range(1997, 2004), seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(f"{years[0]}-01-01", f"{years[-1]}-12-31", freq="D")
    t = np.arange(len(dates))
    weather = pd.DataFrame({
        "date": dates,
        "precip_mm": rng.gamma(0.8, 3.0, len(dates)),
        "tmean_c": 5 + 10 * np.sin(2 * np.pi * t / 365) + rng.normal(0, 2, len(dates)),
    })
    sample_dates = pd.date_range("1998-01-15", f"{years[-1]}-12-31", freq="14D")
    acidity = pd.DataFrame({
        "station": "BD", "date": sample_dates,
        "acidity_mgL": 15000 + 4000 * np.sin(2 * np.pi * np.arange(len(sample_dates)) / 26)
                       + rng.normal(0, 800, len(sample_dates)),
    })
    return acidity, weather


# --- folds ----------------------------------------------------------------

def test_every_year_is_held_out_exactly_once():
    years = list(range(2000, 2014))
    folds = year_folds(years, 5, seed=42)
    flat = [y for f in folds for y in f]
    assert sorted(flat) == years
    assert max(map(len, folds)) - min(map(len, folds)) <= 1
    assert year_folds(years, 5, seed=42) == folds
    assert year_folds(years, 5, seed=7) != folds
    with pytest.raises(ValueError):
        year_folds([2000, 2001], 3, seed=1)


def test_validation_years_never_overlap_the_held_out_years():
    sample_years = np.repeat(np.arange(2000, 2010), 5)
    labels = fold_labels(sample_years, [2003, 2007], n_val_years=2, seed=1)
    by_year = pd.Series(labels).groupby(sample_years).agg(lambda s: set(s))
    assert all(v == {"test"} for y, v in by_year.items() if y in (2003, 2007))
    val_years = [y for y, v in by_year.items() if v == {"val"}]
    assert len(val_years) == 2 and not set(val_years) & {2003, 2007}
    assert all(len(v) == 1 for v in by_year)         # whole years, never split


# --- recipes --------------------------------------------------------------

def test_recipes_survive_their_tags(cfg):
    r = candidate_recipe(cfg, "champion")
    assert r == Recipe("B", 10, "none", "ensemble", 5)
    assert recipe_from_tags({"recipe_json": r.to_json()}, 5) == r
    assert candidate_recipe(cfg, "challenger").time_tag == "per_step"


def test_the_first_m2_versions_read_as_the_paper_recipe():
    """v1 carries only recipe=paper; it was best-of-n Type B H=10 (D-35)."""
    r = recipe_from_tags({"recipe": "paper", "time_tag": "per_step"}, 5)
    assert r == Recipe("B", 10, "per_step", "best_train_val", 5)
    with pytest.raises(ValueError):
        recipe_from_tags({"role": "champion"}, 5)
    with pytest.raises(ValueError):
        Recipe("B", 10, "none", "best_all_samples", 5)


# --- the gate -------------------------------------------------------------

def bt(r, rmse):
    return SimpleNamespace(r=r, rmse_mgL=rmse)


@pytest.mark.parametrize("cand, inc, promote", [
    (bt(0.6, 2500), None, True),                  # first model, clears the floor
    (bt(0.1, 2500), None, False),                 # below the R floor
    (bt(0.6, 2500), bt(0.6, 2500), True),         # unchanged recipe: a refresh
    (bt(0.7, 2400), bt(0.6, 2500), True),         # better on both
    (bt(0.7, 2600), bt(0.6, 2500), False),        # worse RMSE
    (bt(0.55, 2400), bt(0.6, 2500), False),       # R dropped
    (bt(float("nan"), 2400), bt(0.6, 2500), False),
])
def test_gate_promotes_only_a_recipe_that_is_no_worse(cfg, cand, inc, promote):
    assert gate(cand, inc, cfg).promote is promote


# --- the ensemble ---------------------------------------------------------

def test_an_ensemble_averages_its_members_in_normalised_units():
    spec = SimpleNamespace(n_features=2, hidden_size=4, output_activation="linear",
                           freeze_bias_hh=False)
    nets = []
    for seed in range(3):
        torch.manual_seed(seed)
        nets.append(build_network(spec))
    X = np.random.default_rng(0).normal(size=(7, 10, 2)).astype(np.float32)
    np.testing.assert_allclose(predict_members(nets, X),
                               np.mean([predict(n, X) for n in nets], axis=0))
    np.testing.assert_array_equal(predict_members(nets[:1], X), predict(nets[0], X))
    np.testing.assert_array_equal(predict_members(nets[0], X), predict(nets[0], X))


# --- backtest and refit on synthetic data ----------------------------------

@pytest.mark.parametrize("selection", ["ensemble", "best_train_val"])
def test_backtest_gives_each_sample_one_out_of_fold_prediction(cfg, selection):
    acidity, weather = synthetic_data()
    c = fast(cfg, selection=selection)
    recipe = Recipe("B", 4, "none", selection, 2)
    res = backtest(c, acidity, weather, "BD", recipe)

    assert not np.isnan(res.oof_mgL).any()
    held = sorted(y for f in res.folds for y in f)
    assert held == sorted(set(pd.DatetimeIndex(res.dates).year))
    assert sum(f["n_test"] for f in res.per_fold) == len(res.y_mgL)
    assert len(res.epochs_run) == 3 * 2
    assert np.isfinite(res.r) and res.rmse_mgL > 0


def test_backtest_is_deterministic(cfg):
    acidity, weather = synthetic_data()
    c = fast(cfg)
    recipe = Recipe("B", 4, "none", "ensemble", 2)
    a = backtest(c, acidity, weather, "BD", recipe)
    b = backtest(c, acidity, weather, "BD", recipe)
    np.testing.assert_array_equal(a.oof_mgL, b.oof_mgL)


def test_refit_trains_every_seed_for_exactly_the_given_epochs(cfg):
    acidity, weather = synthetic_data()
    c = fast(cfg)
    members, scaler, s = refit(c, acidity, weather, "BD", Recipe("B", 4, "none", "ensemble", 2), 5)
    assert len(members) == 2
    assert set(s.meta["split"]) == {"train"}
    assert scaler.n_fit == len(s)

    one, _, _ = refit(c, acidity, weather, "BD", Recipe("B", 4, "none", "best_train_val", 2), 5)
    assert len(one) == 1


# --- the whole flow, into a throwaway local registry -----------------------

@pytest.mark.integration
def test_train_and_register_promotes_then_refreshes(tmp_path):
    """First run: no incumbent, so promote. Second run with the same recipe:
    no worse, so promote again as a refresh. Each version is an ensemble."""
    import mlflow

    from acidity_lstm.registry import train_and_register

    cfg = fast(load_config())
    previous = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(tmp_path.as_uri())
    mlflow.set_experiment("m3-test")
    try:
        first = train_and_register(cfg, stations=["BD"], roles=["champion"], mlflow=mlflow)
        second = train_and_register(cfg, stations=["BD"], roles=["champion"], mlflow=mlflow)
        client = mlflow.MlflowClient()
        name = first["name"].iloc[0]
        mv = client.get_model_version_by_alias(name, "champion")
        model = mlflow.pyfunc.load_model(f"models:/{name}@champion")
        spec = json.loads(Path(model._model_impl.context.artifacts["spec"]).read_text())
    finally:
        mlflow.set_tracking_uri(previous)

    assert bool(first["promoted"].iloc[0]) and first["version"].iloc[0] == 1
    assert bool(second["promoted"].iloc[0]) and second["version"].iloc[0] == 2
    assert second["inc_r"].iloc[0] == pytest.approx(second["cand_r"].iloc[0])
    assert int(mv.version) == 2 and mv.tags["gate"] == "promoted"
    assert Recipe.from_json(mv.tags["recipe_json"]).selection == "ensemble"
    assert spec["n_members"] == 2
