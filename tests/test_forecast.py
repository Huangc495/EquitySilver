"""Forecast and sensitivity tests (CLAUDE.md Phases 9-10)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from acidity_lstm.config import load_config
from acidity_lstm.experiments import override
from acidity_lstm.forecast import (
    PREDICT_SPLIT,
    build_forecast_samples,
    combined_mse,
    perturb_weather,
    run_forecast,
    run_sensitivity,
)
from acidity_lstm.preprocess import clean_acidity, clean_weather


@pytest.fixture(scope="module")
def data():
    cfg = load_config()
    weather, _ = clean_weather(cfg)
    acidity, _ = clean_acidity(cfg)
    return cfg, acidity, weather


def fast(cfg, **sections):
    base = {"train": {"max_epochs": 5, "patience": 3, "n_repeats": 3}}
    base.update(sections)
    return override(cfg, base)


@pytest.fixture(scope="module")
def forecast(data):
    """The real Phase 9 model, not a shortened one.

    The sensitivity directions are a property of a properly trained network,
    so a truncated training budget would not test what we care about. The full
    run is only a few seconds on 161 samples.
    """
    cfg, acidity, weather = data
    return cfg, run_forecast(cfg, acidity, weather, mlflow=None)


# --- Sample construction --------------------------------------------------

def test_forecast_split_is_time_based(data):
    cfg, acidity, weather = data
    s, _ = build_forecast_samples(cfg, acidity, weather, seed=42)
    years = s.meta["date"].dt.year.to_numpy()
    split = s.meta["split"].to_numpy()

    tr0, tr1 = cfg["forecast"]["train_years"]
    p0, p1 = cfg["forecast"]["predict_years"]

    assert set(years[split == PREDICT_SPLIT]) <= set(range(p0, p1 + 1))
    for name in ("train", "val"):
        assert set(years[split == name]) <= set(range(tr0, tr1 + 1))


def test_forecast_counts_match_phase2(data):
    """161 training-period samples and 24 forecast samples (Phase 2 report)."""
    cfg, acidity, weather = data
    s, _ = build_forecast_samples(cfg, acidity, weather, seed=42)
    counts = s.meta["split"].value_counts()
    assert counts["train"] + counts["val"] == 161
    assert counts[PREDICT_SPLIT] == 24


def test_train_validation_ratio_is_80_20(data):
    cfg, acidity, weather = data
    s, _ = build_forecast_samples(cfg, acidity, weather, seed=42)
    counts = s.meta["split"].value_counts()
    n = counts["train"] + counts["val"]
    assert counts["train"] == round(0.8 * n)
    assert counts["val"] == n - counts["train"]


def test_no_held_out_test_inside_the_training_period(data):
    """CLAUDE.md: 80/20 train/validation, no test set within 1999-2014."""
    cfg, acidity, weather = data
    s, _ = build_forecast_samples(cfg, acidity, weather, seed=42)
    tr0, tr1 = cfg["forecast"]["train_years"]
    years = s.meta["date"].dt.year.to_numpy()
    in_train_period = (years >= tr0) & (years <= tr1)
    assert not (s.meta["split"].to_numpy()[in_train_period] == PREDICT_SPLIT).any()


def test_1998_and_2017_are_excluded(data):
    cfg, acidity, weather = data
    s, _ = build_forecast_samples(cfg, acidity, weather, seed=42)
    years = set(s.meta["date"].dt.year)
    assert 1998 not in years and 2017 not in years


def test_samples_carry_the_time_tag(data):
    cfg, acidity, weather = data
    s, _ = build_forecast_samples(cfg, acidity, weather, seed=42)
    assert s.X.shape[2] == 3
    assert s.feature_names[2] == "day_number"


# --- The scaler must not see the forecast period --------------------------

def test_scaler_is_fitted_on_the_training_period_only(data):
    cfg, acidity, weather = data
    s, scaler = build_forecast_samples(cfg, acidity, weather, seed=42)
    assert scaler.n_fit == 161
    assert scaler.fit_on == "train_period"

    train_mask = (s.meta["split"] != PREDICT_SPLIT).to_numpy()
    assert scaler.y_mean == pytest.approx(float(s.y[train_mask].mean()), rel=1e-5)
    # Fitting on everything would give a different mean, so this is a real check.
    assert not np.isclose(scaler.y_mean, float(s.y.mean()))


def test_time_tag_extrapolates_beyond_the_training_range(data):
    """The caveat CLAUDE.md asks to be reported."""
    cfg, acidity, weather = data
    s, scaler = build_forecast_samples(cfg, acidity, weather, seed=42)
    is_predict = (s.meta["split"] == PREDICT_SPLIT).to_numpy()

    train_max = s.meta.loc[~is_predict, "day_number"].max()
    predict_min = s.meta.loc[is_predict, "day_number"].min()
    assert predict_min > train_max, "forecast tags must lie beyond training"

    z = (s.X[is_predict, :, 2] - scaler.x_mean[2]) / scaler.x_std[2]
    z_train_max = (train_max - scaler.x_mean[2]) / scaler.x_std[2]
    assert z.max() > z_train_max


# --- Selection ------------------------------------------------------------

def test_best_is_selected_on_train_plus_validation_not_the_forecast(forecast):
    """Selecting on the forecast would leak the answer."""
    cfg, fc = forecast
    scores = [fc.train_val_mse(r) for r in fc.results]
    assert fc.train_val_mse() == pytest.approx(min(scores))


def test_combined_mse_is_sample_weighted():
    from acidity_lstm.evaluate import Evaluation, SplitMetrics

    ev = Evaluation()
    ev.by_split = {
        "train": SplitMetrics(n=80, mse=0.5, r=0.0, rmse_mgL=0.0),
        "val": SplitMetrics(n=20, mse=1.0, r=0.0, rmse_mgL=0.0),
    }
    assert combined_mse(ev, ("train", "val")) == pytest.approx(0.6)


def test_combined_mse_ignores_empty_splits():
    from acidity_lstm.evaluate import Evaluation, SplitMetrics

    ev = Evaluation()
    ev.by_split = {
        "train": SplitMetrics(n=10, mse=0.4, r=0.0, rmse_mgL=0.0),
        "val": SplitMetrics(n=0, mse=float("nan"), r=0.0, rmse_mgL=0.0),
    }
    assert combined_mse(ev, ("train", "val")) == pytest.approx(0.4)


def test_forecast_reports_a_spread(forecast):
    cfg, fc = forecast
    sp = fc.spread()
    assert len(sp["values"]) == cfg["train"]["n_repeats"]
    assert sp["min"] <= sp["mean"] <= sp["max"]
    assert np.isfinite(sp["sd"])


def test_forecast_metrics_are_finite(forecast):
    cfg, fc = forecast
    assert np.isfinite(fc.train_val_mse())
    assert np.isfinite(fc.predict_mse())
    assert fc.predict_rmse_mgL() > 0


# --- Perturbation ---------------------------------------------------------

def test_perturb_scales_only_inside_the_range(data):
    cfg, acidity, weather = data
    out = perturb_weather(weather, "precip", 1.2, "2015-01-01", "2016-12-31")

    inside = out["date"].between("2015-01-01", "2016-12-31")
    a = weather.loc[inside, "precip_mm"].to_numpy()
    b = out.loc[inside, "precip_mm"].to_numpy()
    assert np.allclose(b, a * 1.2, equal_nan=True)

    outside = ~inside
    assert np.allclose(
        out.loc[outside, "precip_mm"].to_numpy(),
        weather.loc[outside, "precip_mm"].to_numpy(),
        equal_nan=True,
    )


def test_perturb_leaves_the_other_variable_alone(data):
    cfg, acidity, weather = data
    out = perturb_weather(weather, "precip", 0.8, "2015-01-01", "2016-12-31")
    assert np.allclose(out["tmean_c"].to_numpy(),
                       weather["tmean_c"].to_numpy(), equal_nan=True)


def test_perturb_with_factor_one_is_the_identity(data):
    cfg, acidity, weather = data
    out = perturb_weather(weather, "tmean", 1.0, "2015-01-01", "2016-12-31")
    assert np.allclose(out["tmean_c"].to_numpy(),
                       weather["tmean_c"].to_numpy(), equal_nan=True)


def test_perturb_does_not_mutate_the_input(data):
    cfg, acidity, weather = data
    before = weather["precip_mm"].to_numpy().copy()
    perturb_weather(weather, "precip", 1.2, "2015-01-01", "2016-12-31")
    assert np.allclose(weather["precip_mm"].to_numpy(), before, equal_nan=True)


def test_temperature_scaling_widens_the_swing_both_ways(data):
    """Scaling degrees C amplifies summer highs and winter lows together."""
    cfg, acidity, weather = data
    hot = perturb_weather(weather, "tmean", 1.2, "2015-01-01", "2016-12-31")
    inside = hot["date"].between("2015-01-01", "2016-12-31")

    base = weather.loc[inside, "tmean_c"].dropna()
    wide = hot.loc[inside, "tmean_c"].dropna()
    assert wide.max() > base.max()
    assert wide.min() < base.min()


def test_unknown_variable_raises(data):
    cfg, acidity, weather = data
    with pytest.raises(ValueError, match="Unknown variable"):
        perturb_weather(weather, "humidity", 1.2, "2015-01-01", "2016-12-31")


# --- Sensitivity ----------------------------------------------------------

@pytest.fixture(scope="module")
def sensitivity(data, forecast):
    _, acidity, weather = data
    cfg, fc = forecast
    return run_sensitivity(cfg, acidity, weather, fc, mlflow=None)


def test_sensitivity_covers_every_scenario(sensitivity):
    table, curves, meta = sensitivity
    assert len(table) == 5                       # baseline + 2 variables x 2 factors
    assert set(table["scenario"]) == set(curves)
    assert table.loc[0, "scenario"] == "Real weather"
    assert table.loc[0, "mean_change_mgL"] == 0.0


def test_sensitivity_curves_align_with_the_same_samples(sensitivity):
    table, curves, meta = sensitivity
    lengths = {len(v) for v in curves.values()}
    assert lengths == {len(meta)} == {24}


def test_perturbation_changes_the_predictions(sensitivity):
    table, curves, meta = sensitivity
    base = curves["Real weather"]
    for label, values in curves.items():
        if label == "Real weather":
            continue
        assert not np.allclose(values, base), f"{label} had no effect"


def test_sensitivity_directions_match_the_paper(sensitivity):
    """More precipitation lowers acidity; a wider temperature swing raises it."""
    table, _, _ = sensitivity
    precip = table[table["variable"] == "precip"].set_index("factor")
    tmean = table[table["variable"] == "tmean"].set_index("factor")

    assert precip.loc[1.2, "mean_change_mgL"] < 0 < precip.loc[0.8, "mean_change_mgL"]
    assert tmean.loc[1.2, "mean_change_mgL"] > 0 > tmean.loc[0.8, "mean_change_mgL"]


def test_temperature_is_the_more_sensitive_input(sensitivity):
    table, _, _ = sensitivity
    precip = table[table["variable"] == "precip"]["mean_change_pct"].abs().sum()
    tmean = table[table["variable"] == "tmean"]["mean_change_pct"].abs().sum()
    assert tmean > precip


def test_sensitivity_predictions_are_in_mgL(sensitivity):
    table, curves, _ = sensitivity
    for values in curves.values():
        assert np.all(np.asarray(values) > 1000), "predictions should be in mg/L"
        assert np.all(np.asarray(values) < 60000)
