"""Scaling tests required by CLAUDE.md Phase 3."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from acidity_lstm.config import Config, load_config
from acidity_lstm.scaling import Scaler, fit_scaler, fit_scaler_for
from acidity_lstm.windows import SampleSet


@pytest.fixture
def cfg():
    return load_config()


def with_scaling(cfg: Config, **updates) -> Config:
    import copy

    raw = copy.deepcopy(cfg.raw)
    raw["scaling"].update(updates)
    return Config(raw=raw, paths=cfg.paths, repo_root=cfg.repo_root)


def make_sample_set(n=20, n_steps=10, n_features=2, seed=0, splits=None):
    rng = np.random.default_rng(seed)
    X = rng.normal(5, 3, (n, n_steps, n_features)).astype(np.float32)
    y = rng.normal(13000, 3000, n).astype(np.float32)
    if splits is None:
        splits = ["train"] * (n - 6) + ["val"] * 3 + ["test"] * 3
    meta = pd.DataFrame(
        {
            "station": "BD",
            "date": pd.date_range("2000-01-01", periods=n, freq="14D"),
            "acidity_mgL": y.astype(float),
            "day_number": np.arange(n) * 14 + 1,
            "split": splits,
        }
    )
    names = ("precip", "tmean", "day_number")[:n_features]
    return SampleSet(
        X=X, y=y, meta=meta, station="BD", window_type="B",
        feature_names=names, n_candidates=n, n_dropped=0,
    )


# --- Core z-score behaviour ----------------------------------------------

def test_zscore_uses_population_std_by_default():
    """Eq. 9 is misprinted; population std (ddof = 0) is the convention."""
    X = np.arange(24, dtype=float).reshape(2, 4, 3)
    y = np.array([1.0, 2.0])
    s = fit_scaler(X, y, ("a", "b", "c"), ddof=0)

    assert np.allclose(s.x_mean, X.mean(axis=(0, 1)))
    assert np.allclose(s.x_std, X.std(axis=(0, 1), ddof=0))
    assert s.y_std == pytest.approx(y.std(ddof=0))
    # ddof=1 would differ, so the convention is not accidental.
    assert not np.isclose(s.x_std[0], X[:, :, 0].std(ddof=1))


def test_sample_std_option_is_honoured():
    X = np.arange(24, dtype=float).reshape(2, 4, 3)
    y = np.array([1.0, 5.0])
    s = fit_scaler(X, y, ("a", "b", "c"), ddof=1)
    assert np.allclose(s.x_std, X.std(axis=(0, 1), ddof=1))


def test_one_statistic_per_feature_pooled_over_all_steps():
    """Statistics pool over samples AND time steps, giving one pair per feature."""
    X = np.zeros((5, 10, 2))
    X[:, :, 0] = 100.0                  # constant-ish feature with a single outlier
    X[0, 0, 0] = 200.0
    X[:, :, 1] = np.arange(50).reshape(5, 10)
    s = fit_scaler(X, np.ones(5), ("precip", "tmean"))

    assert s.x_mean.shape == (2,)
    assert s.x_std.shape == (2,)
    assert s.x_mean[0] == pytest.approx(X[:, :, 0].mean())
    assert s.x_mean[1] == pytest.approx(np.arange(50).mean())


def test_transform_produces_zero_mean_unit_variance():
    rng = np.random.default_rng(0)
    X = rng.normal(7, 2, (30, 10, 2))
    y = rng.normal(13000, 2000, 30)
    s = fit_scaler(X, y, ("precip", "tmean"))

    Xn, yn = s.transform(X, y)
    assert np.allclose(Xn.mean(axis=(0, 1)), 0, atol=1e-5)
    assert np.allclose(Xn.std(axis=(0, 1)), 1, atol=1e-5)
    assert yn.mean() == pytest.approx(0, abs=1e-5)
    assert yn.std() == pytest.approx(1, abs=1e-5)


def test_inverse_transform_round_trips():
    rng = np.random.default_rng(1)
    X = rng.normal(0, 1, (12, 10, 2))
    y = rng.normal(13000, 3000, 12)
    s = fit_scaler(X, y, ("precip", "tmean"))
    assert np.allclose(s.inverse_transform_y(s.transform_y(y)), y, rtol=1e-4)


def test_inverse_transform_recovers_mgL_units():
    s = fit_scaler(np.zeros((4, 10, 2)), np.array([10.0, 20.0, 30.0, 40.0]), ("a", "b"))
    assert s.inverse_transform_y(np.array([0.0])) == pytest.approx(25.0)
    assert s.inverse_transform_y(s.transform_y(np.array([40.0])))[0] == pytest.approx(40.0)


def test_three_features_are_scaled_independently():
    rng = np.random.default_rng(2)
    X = rng.normal(0, 1, (20, 10, 3))
    X[:, :, 2] = np.linspace(1, 7000, 200).reshape(20, 10)   # the time tag
    s = fit_scaler(X, rng.normal(0, 1, 20), ("precip", "tmean", "day_number"))

    assert len(s.x_mean) == 3
    assert s.x_mean[2] == pytest.approx(X[:, :, 2].mean())
    Xn = s.transform_x(X)
    assert np.allclose(Xn.std(axis=(0, 1)), 1, atol=1e-5)


# --- Guards ---------------------------------------------------------------

def test_constant_feature_does_not_divide_by_zero():
    X = np.ones((10, 10, 2))
    X[:, :, 1] = np.arange(100).reshape(10, 10)
    s = fit_scaler(X, np.arange(10, dtype=float), ("precip", "tmean"))
    Xn = s.transform_x(X)
    assert s.x_std[0] == 1.0
    assert np.isfinite(Xn).all()


def test_constant_output_does_not_divide_by_zero():
    s = fit_scaler(np.random.rand(5, 10, 2), np.full(5, 12000.0), ("a", "b"))
    assert s.y_std == 1.0
    assert np.isfinite(s.transform_y(np.full(5, 12000.0))).all()


def test_feature_count_mismatch_raises():
    s = fit_scaler(np.random.rand(5, 10, 2), np.arange(5, dtype=float), ("a", "b"))
    with pytest.raises(ValueError, match="features"):
        s.transform_x(np.random.rand(5, 10, 3))


def test_empty_fitting_set_raises():
    with pytest.raises(ValueError, match="empty"):
        fit_scaler(np.empty((0, 10, 2)), np.empty(0), ("a", "b"))


def test_wrong_input_rank_raises():
    with pytest.raises(ValueError, match="shape"):
        fit_scaler(np.random.rand(5, 2), np.arange(5, dtype=float), ("a", "b"))


# --- Fitting set selection ------------------------------------------------

def test_fit_on_all_uses_every_sample(cfg):
    s = make_sample_set(n=20)
    scaler = fit_scaler_for(s, with_scaling(cfg, fit_on="all"))
    assert scaler.n_fit == 20
    assert scaler.fit_on == "all"
    assert scaler.y_mean == pytest.approx(float(s.y.mean()), rel=1e-5)


def test_fit_on_train_uses_only_training_rows(cfg):
    s = make_sample_set(n=20)
    scaler = fit_scaler_for(s, with_scaling(cfg, fit_on="train"))
    train_mask = s.meta["split"].to_numpy() == "train"

    assert scaler.n_fit == int(train_mask.sum()) == 14
    assert scaler.y_mean == pytest.approx(float(s.y[train_mask].mean()), rel=1e-5)


def test_fit_on_all_and_train_give_different_parameters(cfg):
    s = make_sample_set(n=40, seed=7)
    a = fit_scaler_for(s, with_scaling(cfg, fit_on="all"))
    t = fit_scaler_for(s, with_scaling(cfg, fit_on="train"))
    assert a.y_mean != t.y_mean, "the leak from fit_on='all' should be visible"


def test_fit_on_train_without_a_training_split_raises(cfg):
    s = make_sample_set(n=6, splits=["test"] * 6)
    with pytest.raises(ValueError, match="training split"):
        fit_scaler_for(s, with_scaling(cfg, fit_on="train"))


def test_unknown_fit_on_raises(cfg):
    with pytest.raises(ValueError, match="fit_on"):
        fit_scaler_for(make_sample_set(), with_scaling(cfg, fit_on="validation"))


def test_scaler_records_its_scenario(cfg):
    s = make_sample_set()
    scaler = fit_scaler_for(s, cfg)
    assert scaler.station == "BD"
    assert scaler.window_type == "B"


# --- Logging --------------------------------------------------------------

def test_params_dict_is_flat_and_loggable(cfg):
    scaler = fit_scaler_for(make_sample_set(n_features=3), cfg)
    p = scaler.params

    assert p["scaler_fit_on"] == cfg["scaling"]["fit_on"]
    assert p["scaler_ddof"] == 0
    for name in ("precip", "tmean", "day_number"):
        assert f"scaler_x_mean_{name}" in p
        assert f"scaler_x_std_{name}" in p
    assert all(isinstance(v, (int, float, str)) for v in p.values())


# --- Scope: one scaler per scenario --------------------------------------

def test_separate_scalers_per_station_and_type(cfg):
    bd = make_sample_set(n=20, seed=1)
    c7 = make_sample_set(n=20, seed=2)
    c7.meta["station"] = "C7"
    object.__setattr__(c7, "station", "C7")

    sb = fit_scaler_for(bd, cfg)
    sc = fit_scaler_for(c7, cfg)
    assert sb.y_mean != sc.y_mean
    # Using the wrong station's scaler must not silently look correct.
    assert not np.allclose(sb.transform_y(c7.y), sc.transform_y(c7.y))
