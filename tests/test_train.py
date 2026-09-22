"""Training-loop, metric and end-to-end tests (CLAUDE.md Phase 5)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from acidity_lstm.config import Config, load_config
from acidity_lstm.evaluate import evaluate_model, mse, pearson_r, predict, rmse
from acidity_lstm.models import set_seed
from acidity_lstm.scaling import fit_scaler_for
from acidity_lstm.train import best_of, summarise_repeats, train_model, train_repeats
from acidity_lstm.windows import SampleSet


@pytest.fixture
def cfg():
    return load_config()


def override(cfg: Config, **sections) -> Config:
    import copy

    raw = copy.deepcopy(cfg.raw)
    for section, updates in sections.items():
        raw[section].update(updates)
    return Config(raw=raw, paths=cfg.paths, repo_root=cfg.repo_root)


def learnable_set(n=120, n_steps=10, n_features=2, seed=0, noise=0.05):
    """A set where the target is a clean function of the inputs."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, n_steps, n_features)).astype(np.float32)
    signal = X[:, -1, 0] * 2.0 + X[:, :, 1].mean(axis=1) - 0.5 * X[:, 0, 0]
    y = (13000 + 2000 * (signal + rng.normal(0, noise, n))).astype(np.float32)

    sizes = [int(n * 0.7), int(n * 0.15)]
    labels = np.array(
        ["train"] * sizes[0] + ["val"] * sizes[1] + ["test"] * (n - sum(sizes)),
        dtype=object,
    )
    rng.shuffle(labels)
    meta = pd.DataFrame(
        {
            "station": "BD",
            "date": pd.date_range("2000-01-01", periods=n, freq="14D"),
            "acidity_mgL": y.astype(float),
            "day_number": np.arange(n) * 14 + 1,
            "split": labels,
        }
    )
    return SampleSet(
        X=X, y=y, meta=meta, station="BD", window_type="B",
        feature_names=("precip", "tmean", "day_number")[:n_features],
        n_candidates=n, n_dropped=0,
    )


# --- Metrics --------------------------------------------------------------

def test_mse_and_rmse():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.0, 2.0, 5.0])
    assert mse(a, b) == pytest.approx(4 / 3)
    assert rmse(a, b) == pytest.approx(np.sqrt(4 / 3))
    assert mse(a, a) == 0.0


def test_pearson_r_matches_numpy():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=50), rng.normal(size=50)
    assert pearson_r(a, b) == pytest.approx(np.corrcoef(a, b)[0, 1])


def test_pearson_r_edge_cases():
    a = np.array([1.0, 2.0, 3.0])
    assert pearson_r(a, 2 * a + 1) == pytest.approx(1.0)
    assert pearson_r(a, -a) == pytest.approx(-1.0)
    assert np.isnan(pearson_r(a, np.ones(3))), "constant series -> undefined"
    assert np.isnan(pearson_r(np.array([1.0]), np.array([1.0])))


def test_mse_on_empty_is_nan():
    assert np.isnan(mse(np.empty(0), np.empty(0)))


# --- Evaluation -----------------------------------------------------------

def test_evaluate_reports_every_split(cfg):
    s = learnable_set(n=60)
    scaler = fit_scaler_for(s, cfg)
    set_seed(0)
    res = train_model(s, scaler, override(cfg, train={"max_epochs": 5}), seed=0, hidden_size=5)
    ev = res.evaluation

    assert set(ev.by_split) == {"all", "train", "val", "test"}
    assert ev.by_split["all"].n == 60
    assert sum(ev.by_split[k].n for k in ("train", "val", "test")) == 60
    for m in ev.by_split.values():
        assert np.isfinite(m.mse) and np.isfinite(m.rmse_mgL)


def test_rmse_is_reported_in_mgL(cfg):
    """RMSE must be on the physical scale, not the normalised one."""
    s = learnable_set(n=60)
    scaler = fit_scaler_for(s, cfg)
    res = train_model(s, scaler, override(cfg, train={"max_epochs": 3}), seed=0, hidden_size=5)
    m = res.evaluation.all
    assert m.rmse_mgL == pytest.approx(np.sqrt(m.mse) * scaler.y_std, rel=1e-4)
    assert m.rmse_mgL > 100, "mg/L RMSE should be on the order of thousands"


def test_predictions_are_finite_and_right_length(cfg):
    s = learnable_set(n=40)
    scaler = fit_scaler_for(s, cfg)
    res = train_model(s, scaler, override(cfg, train={"max_epochs": 3}), seed=0, hidden_size=5)
    p = predict(res.model, scaler.transform_x(s.X))
    assert p.shape == (40,)
    assert np.isfinite(p).all()


# --- Training behaviour ---------------------------------------------------

def test_training_reduces_error_on_a_learnable_problem(cfg):
    s = learnable_set(n=150, noise=0.02)
    scaler = fit_scaler_for(s, cfg)
    c = override(cfg, train={"max_epochs": 200, "patience": 20})

    set_seed(0)
    res = train_model(s, scaler, c, seed=0, hidden_size=20)
    assert res.evaluation.all.mse < 0.5, "should learn a clean signal"
    assert res.evaluation.all.r > 0.7


def test_early_stopping_triggers_on_a_flat_problem(cfg):
    """Pure noise: validation never improves, so patience must fire."""
    rng = np.random.default_rng(0)
    s = learnable_set(n=80)
    s.y[:] = rng.normal(13000, 3000, 80).astype(np.float32)
    s.meta["acidity_mgL"] = s.y.astype(float)

    scaler = fit_scaler_for(s, cfg)
    c = override(cfg, train={"max_epochs": 200, "patience": 6})
    res = train_model(s, scaler, c, seed=0, hidden_size=5)

    assert res.stopped_early
    assert res.epochs_run < 200
    assert res.epochs_run - res.best_epoch >= 6


def test_max_epochs_is_respected(cfg):
    s = learnable_set(n=80)
    scaler = fit_scaler_for(s, cfg)
    c = override(cfg, train={"max_epochs": 7, "patience": 1000})
    res = train_model(s, scaler, c, seed=0, hidden_size=5)
    assert res.epochs_run == 7
    assert not res.stopped_early
    assert len(res.history["epoch"]) == 7


def test_learning_rate_schedule_drops_after_the_period(cfg):
    s = learnable_set(n=80)
    scaler = fit_scaler_for(s, cfg)
    c = override(cfg, train={"max_epochs": 6, "patience": 1000,
                             "lr_drop_period": 3, "lr_drop_factor": 0.2})
    res = train_model(s, scaler, c, seed=0, hidden_size=5)
    lrs = res.history["lr"]
    assert lrs[0] == pytest.approx(0.005)
    assert lrs[-1] == pytest.approx(0.005 * 0.2 ** 2, rel=1e-6)


def test_restore_best_returns_the_best_epoch(cfg):
    s = learnable_set(n=100)
    scaler = fit_scaler_for(s, cfg)
    c = override(cfg, train={"max_epochs": 60, "patience": 6, "restore_best": True})
    res = train_model(s, scaler, c, seed=0, hidden_size=10)

    from acidity_lstm.train import _subset_mse

    X = torch.as_tensor(scaler.transform_x(s.X))
    y = torch.as_tensor(scaler.transform_y(s.y))
    val_idx = np.flatnonzero(s.meta["split"].to_numpy() == "val")
    assert _subset_mse(res.model, X, y, val_idx) == pytest.approx(res.best_val_mse, rel=1e-5)


def test_default_returns_the_final_network_not_the_best(cfg):
    """MATLAB behaviour, D-12: restore_best is false by default.

    Uses a target the model cannot fit, so early stopping is guaranteed to
    fire and the final network is provably not the best one seen.
    """
    assert cfg["train"]["restore_best"] is False

    rng = np.random.default_rng(3)
    s = learnable_set(n=100)
    s.y[:] = rng.normal(13000, 3000, 100).astype(np.float32)
    s.meta["acidity_mgL"] = s.y.astype(float)

    scaler = fit_scaler_for(s, cfg)
    c = override(cfg, train={"max_epochs": 200, "patience": 6, "restore_best": False})
    res = train_model(s, scaler, c, seed=0, hidden_size=10)

    assert res.stopped_early
    assert res.best_epoch < res.epochs_run, "stopping implies the best was earlier"

    from acidity_lstm.train import _subset_mse

    X = torch.as_tensor(scaler.transform_x(s.X))
    y = torch.as_tensor(scaler.transform_y(s.y))
    val_idx = np.flatnonzero(s.meta["split"].to_numpy() == "val")
    final_val = _subset_mse(res.model, X, y, val_idx)

    # The returned network is the last one, which by construction never beat
    # the best; restore_best=True would have returned best_val_mse instead.
    assert final_val >= res.best_val_mse


def test_drop_last_discards_the_partial_batch(cfg):
    """25 training rows at batch 20 -> one batch of 20, 5 discarded."""
    from acidity_lstm.train import _iter_batches

    order = np.arange(25)
    batches = list(_iter_batches(25, 20, order, drop_last=True))
    assert len(batches) == 1 and len(batches[0]) == 20

    batches = list(_iter_batches(25, 20, order, drop_last=False))
    assert len(batches) == 2 and len(batches[1]) == 5


def test_drop_last_does_not_train_on_nothing():
    """Fewer rows than one batch must still produce a batch."""
    from acidity_lstm.train import _iter_batches

    batches = list(_iter_batches(12, 20, np.arange(12), drop_last=True))
    assert len(batches) == 1 and len(batches[0]) == 12


def test_unknown_shuffle_raises(cfg):
    s = learnable_set(n=40)
    scaler = fit_scaler_for(s, cfg)
    with pytest.raises(ValueError, match="shuffle"):
        train_model(s, scaler, override(cfg, train={"shuffle": "sorted"}), seed=0, hidden_size=5)


def test_empty_training_split_raises(cfg):
    s = learnable_set(n=40)
    s.meta["split"] = "test"
    scaler = fit_scaler_for(s, override(cfg, scaling={"fit_on": "all"}))
    with pytest.raises(ValueError, match="Training split is empty"):
        train_model(s, scaler, cfg, seed=0, hidden_size=5)


def test_frozen_bias_hh_is_excluded_from_the_optimizer(cfg):
    s = learnable_set(n=60)
    scaler = fit_scaler_for(s, cfg)
    c = override(cfg, model={"freeze_bias_hh": True}, train={"max_epochs": 5})
    res = train_model(s, scaler, c, seed=0, hidden_size=10)
    assert torch.equal(res.model.lstm.bias_hh_l0, torch.zeros(40))


# --- Reproducibility and repeats -----------------------------------------

def test_same_seed_reproduces_the_run(cfg):
    s = learnable_set(n=80)
    scaler = fit_scaler_for(s, cfg)
    c = override(cfg, train={"max_epochs": 15, "patience": 1000})

    a = train_model(s, scaler, c, seed=7, hidden_size=10)
    b = train_model(s, scaler, c, seed=7, hidden_size=10)
    assert a.evaluation.all.mse == pytest.approx(b.evaluation.all.mse, rel=1e-9)
    assert a.history["train_loss"] == pytest.approx(b.history["train_loss"], rel=1e-9)


def test_different_seeds_diverge(cfg):
    s = learnable_set(n=80)
    scaler = fit_scaler_for(s, cfg)
    c = override(cfg, train={"max_epochs": 15, "patience": 1000})
    a = train_model(s, scaler, c, seed=1, hidden_size=10)
    b = train_model(s, scaler, c, seed=2, hidden_size=10)
    assert a.evaluation.all.mse != b.evaluation.all.mse


def test_repeats_vary_only_the_initialisation(cfg):
    s = learnable_set(n=90)
    scaler = fit_scaler_for(s, cfg)
    c = override(cfg, train={"max_epochs": 12, "patience": 1000, "n_repeats": 3})
    results = train_repeats(s, scaler, c, hidden_size=10)

    assert len(results) == 3
    assert [r.seed for r in results] == [42, 43, 44]
    # The split is fixed across repeats.
    for r in results:
        assert r.evaluation.by_split["test"].n == results[0].evaluation.by_split["test"].n


def test_best_of_selects_lowest_all_sample_mse(cfg):
    s = learnable_set(n=90)
    scaler = fit_scaler_for(s, cfg)
    c = override(cfg, train={"max_epochs": 12, "patience": 1000, "n_repeats": 3})
    results = train_repeats(s, scaler, c, hidden_size=10)

    best = best_of(results)
    assert best.selection_mse == min(r.selection_mse for r in results)
    assert best.selection_mse == best.evaluation.all.mse


def test_summarise_repeats_reports_mean_sd_and_best(cfg):
    s = learnable_set(n=90)
    scaler = fit_scaler_for(s, cfg)
    c = override(cfg, train={"max_epochs": 12, "patience": 1000, "n_repeats": 3})
    summary = summarise_repeats(train_repeats(s, scaler, c, hidden_size=10))

    assert summary["n_repeats"] == 3
    for key in ("all_mse_mean", "all_mse_sd", "best_all_mse", "best_seed",
                "test_r_mean", "epochs_mean"):
        assert key in summary
    assert summary["best_all_mse"] <= summary["all_mse_mean"] + 1e-9


# --- End to end on the real data -----------------------------------------

def test_end_to_end_on_real_bd_type_b(cfg):
    """A short real run: the pipeline holds together and learns something."""
    from acidity_lstm.preprocess import clean_acidity, clean_weather
    from acidity_lstm.splits import assign_splits
    from acidity_lstm.windows import build_sample_sets

    weather, _ = clean_weather(cfg)
    acidity, _ = clean_acidity(cfg)
    splits = {"BD": assign_splits(acidity, cfg, seed=42, station="BD")}
    sets = build_sample_sets(acidity, weather, cfg, stations=["BD"], types=["B"],
                             splits_by_station=splits)
    s = sets[("BD", "B")]
    assert len(s) == 185

    scaler = fit_scaler_for(s, cfg)
    c = override(cfg, train={"max_epochs": 30, "patience": 6})
    res = train_model(s, scaler, c, seed=42, hidden_size=10)

    assert np.isfinite(res.evaluation.all.mse)
    assert res.evaluation.all.n == 185
    assert res.evaluation.all.rmse_mgL > 0
    assert res.n_parameters == 571


# --- resplit_each_repeat --------------------------------------------------

def test_resplit_each_repeat_requires_a_rebuild_function(cfg):
    """The option must not be silently ignored."""
    s = learnable_set(n=60)
    scaler = fit_scaler_for(s, cfg)
    c = override(cfg, split={"resplit_each_repeat": True},
                 train={"max_epochs": 3, "n_repeats": 2})
    with pytest.raises(ValueError, match="rebuild_fn"):
        train_repeats(s, scaler, c, hidden_size=5)


def test_resplit_each_repeat_calls_the_rebuild_function(cfg):
    c = override(cfg, split={"resplit_each_repeat": True},
                 train={"max_epochs": 3, "n_repeats": 3, "patience": 1000})
    seen = []

    def rebuild(seed):
        seen.append(seed)
        s = learnable_set(n=60, seed=seed)
        return s, fit_scaler_for(s, c)

    base = learnable_set(n=60)
    results = train_repeats(base, fit_scaler_for(base, c), c,
                            hidden_size=5, rebuild_fn=rebuild)
    assert seen == [42, 43, 44]
    assert len(results) == 3


def test_fixed_split_is_the_default(cfg):
    """Default: the split stays put and only initialisation varies."""
    assert cfg["split"]["resplit_each_repeat"] is False
    c = override(cfg, train={"max_epochs": 3, "n_repeats": 2, "patience": 1000})
    s = learnable_set(n=60)
    results = train_repeats(s, fit_scaler_for(s, c), c, hidden_size=5)
    test_ids = [tuple(r.evaluation.by_split["test"].n for _ in (0,)) for r in results]
    assert len(set(test_ids)) == 1
