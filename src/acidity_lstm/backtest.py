"""Backtest, promotion gate and refit (MLOPS.md M3; decisions D9, D10).

A trained model cannot be compared fairly with one trained on different
rows, so the gate compares *recipes*. Both the candidate and the incumbent
are retrained on the same training years and scored on the same held-out
years: blocked cross-validation by year, in which every eligible year is held
out exactly once, so every sample gets one out-of-fold prediction. The
comparison uses R and RMSE in mg/L, never normalised MSE (D-18).

Inside a fold nothing about the held-out years leaks in. Validation years
come from the remaining years, the scaler is fitted on training rows only,
and a recipe that selects among seeds selects on train and validation rows,
never on the held-out ones.

The winner is then refit on every year for the backtest's median epoch
count, with early stopping off, because no year is left to stop on.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from .config import Config
from .evaluate import pearson_r, rmse
from .experiments import VARIANTS, override
from .models import predict
from .scaling import fit_scaler, fit_scaler_for
from .train import train_repeats
from .windows import SampleSet, build_samples, make_weather_arrays

log = logging.getLogger(__name__)

SELECTIONS = ("ensemble", "best_train_val")


@dataclass(frozen=True)
class Recipe:
    """How a model is made: what the gate compares and the refit follows."""

    window_type: str
    hidden_size: int
    time_tag: str               # "none" | "per_step" | "constant"
    selection: str              # "ensemble" (D10) | "best_train_val" (the paper's best-of-n)
    n_seeds: int

    def __post_init__(self):
        if self.selection not in SELECTIONS:
            raise ValueError(f"Unknown selection {self.selection!r}; expected one of {SELECTIONS}.")

    @property
    def with_time_tag(self) -> bool:
        return self.time_tag != "none"

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "Recipe":
        return cls(**json.loads(text))


def candidate_recipe(cfg: Config, role: str) -> Recipe:
    """The recipe the current config would train for `role`."""
    r = cfg["registry"]["roles"][role]
    g = cfg["gate"]
    return Recipe(window_type=r["window_type"], hidden_size=int(r["hidden_size"]),
                  time_tag=r["time_tag"], selection=g["selection"], n_seeds=int(g["n_seeds"]))


def recipe_from_tags(tags: dict, n_seeds: int) -> Recipe:
    """Reconstruct an incumbent's recipe from its model-version tags.

    Versions from M3 on carry `recipe_json`. The first versions (M2) carry
    only `recipe=paper` and `time_tag`; they were Type B, H=10, best of n
    (D-35).
    """
    if "recipe_json" in tags:
        return Recipe.from_json(tags["recipe_json"])
    if tags.get("recipe") == "paper":
        return Recipe(window_type="B", hidden_size=10, time_tag=tags.get("time_tag", "none"),
                      selection="best_train_val", n_seeds=n_seeds)
    raise ValueError(f"Cannot reconstruct a recipe from tags {sorted(tags)}.")


def recipe_config(cfg: Config, recipe: Recipe) -> Config:
    """The training config for a recipe: the paper settings plus its time tag."""
    rcfg = override(cfg, VARIANTS["paper"])
    if recipe.with_time_tag:
        rcfg = override(rcfg, {"windows": {"time_tag": recipe.time_tag}})
    return rcfg


def sample_set_for(cfg: Config, acidity, weather, station: str, recipe: Recipe) -> SampleSet:
    rcfg = recipe_config(cfg, recipe)
    arrays = weather if not isinstance(weather, pd.DataFrame) else make_weather_arrays(weather)
    return build_samples(acidity, arrays, rcfg, station, recipe.window_type,
                         with_time_tag=recipe.with_time_tag)


# --- folds ----------------------------------------------------------------

def year_folds(years, n_folds: int, seed: int) -> list[list[int]]:
    """Deal the years, shuffled with `seed`, into `n_folds` near-equal folds."""
    years = sorted({int(y) for y in years})
    if len(years) < n_folds:
        raise ValueError(f"{len(years)} years cannot fill {n_folds} folds.")
    shuffled = np.array(years)
    np.random.default_rng(seed).shuffle(shuffled)
    return [sorted(int(y) for y in part) for part in np.array_split(shuffled, n_folds)]


def fold_labels(sample_years: np.ndarray, test_years, n_val_years: int, seed: int) -> np.ndarray:
    """train / val / test labels for one fold; validation years never overlap test."""
    remaining = sorted(set(int(y) for y in sample_years) - set(test_years))
    rng = np.random.default_rng(seed)
    val_years = set(rng.choice(remaining, size=min(n_val_years, len(remaining) - 1),
                               replace=False).tolist())
    return np.array(["test" if y in test_years else "val" if y in val_years else "train"
                     for y in sample_years], dtype=object)


# --- training a recipe ----------------------------------------------------

def train_recipe(sample_set: SampleSet, cfg: Config, recipe: Recipe, scaler=None) -> tuple:
    """Train `recipe` on the set's train rows. Returns (members, results, scaler).

    `members` is what the recipe ships: every seed for an ensemble, the best
    seed on train and validation rows for best-of-n.
    """
    rcfg = recipe_config(cfg, recipe)
    base_seed = int(rcfg["train"]["base_seed"])
    if scaler is None:
        scaler = fit_scaler_for(sample_set, override(rcfg, {"scaling": {"fit_on": "train"}}))
    results = train_repeats(sample_set, scaler, rcfg, hidden_size=recipe.hidden_size,
                            base_seed=base_seed, n_repeats=recipe.n_seeds)
    if recipe.selection == "ensemble":
        return [r.model for r in results], results, scaler

    split = sample_set.meta["split"].to_numpy()
    seen = (split == "train") | (split == "val")
    X_seen = scaler.transform_x(sample_set.X[seen])
    y_seen = scaler.transform_y(sample_set.y[seen])
    scores = [float(np.mean((predict(r.model, X_seen) - y_seen) ** 2)) for r in results]
    return [results[int(np.argmin(scores))].model], results, scaler


def ensemble_predict_mgL(members, scaler, X: np.ndarray) -> np.ndarray:
    z = np.mean(np.stack([predict(m, scaler.transform_x(X)) for m in members]), axis=0)
    return scaler.inverse_transform_y(z)


# --- backtest -------------------------------------------------------------

@dataclass
class BacktestResult:
    recipe: Recipe
    station: str
    folds: list
    oof_mgL: np.ndarray                 # one out-of-fold prediction per sample
    y_mgL: np.ndarray
    dates: np.ndarray
    epochs_run: list = field(default_factory=list)
    per_fold: list = field(default_factory=list)

    @property
    def r(self) -> float:
        return pearson_r(self.y_mgL, self.oof_mgL)

    @property
    def rmse_mgL(self) -> float:
        return rmse(self.y_mgL, self.oof_mgL)

    @property
    def median_epochs(self) -> int:
        return int(np.median(self.epochs_run))

    def metrics(self, prefix: str) -> dict:
        return {f"{prefix}_oof_r": self.r, f"{prefix}_oof_rmse_mgL": self.rmse_mgL,
                f"{prefix}_oof_n": float(len(self.y_mgL)),
                f"{prefix}_median_epochs": float(self.median_epochs)}


def backtest(cfg: Config, acidity, weather, station: str, recipe: Recipe) -> BacktestResult:
    """Blocked cross-validation by year for one station and recipe."""
    g = cfg["gate"]
    seed = int(cfg["train"]["base_seed"])
    s = sample_set_for(cfg, acidity, weather, station, recipe)
    years = s.meta["date"].dt.year.to_numpy()
    folds = year_folds(years, int(g["n_folds"]), seed)

    oof = np.full(len(s), np.nan)
    epochs, per_fold = [], []
    for k, test_years in enumerate(folds):
        labels = fold_labels(years, test_years, int(g["val_years"]), seed + k)
        fs = s.with_splits(labels)
        members, results, scaler = train_recipe(fs, cfg, recipe)
        test = labels == "test"
        oof[test] = ensemble_predict_mgL(members, scaler, fs.X[test])
        epochs.extend(r.epochs_run for r in results)
        per_fold.append({"fold": k, "test_years": test_years, "n_test": int(test.sum()),
                         "r": pearson_r(s.y[test], oof[test]),
                         "rmse_mgL": rmse(s.y[test], oof[test])})
    if np.isnan(oof).any():
        raise RuntimeError("Some samples received no out-of-fold prediction.")
    out = BacktestResult(recipe=recipe, station=station, folds=folds, oof_mgL=oof,
                         y_mgL=s.y.astype(float), dates=s.meta["date"].to_numpy(),
                         epochs_run=epochs, per_fold=per_fold)
    log.info("backtest %s %s: out-of-fold R %.3f, RMSE %.0f mg/L over %d samples",
             station, recipe, out.r, out.rmse_mgL, len(s))
    return out


# --- the gate -------------------------------------------------------------

@dataclass
class GateDecision:
    promote: bool
    reason: str


def gate(candidate: BacktestResult, incumbent: BacktestResult | None, cfg: Config) -> GateDecision:
    """Promote when the candidate recipe is no worse than the incumbent's.

    "No worse" makes re-running an unchanged recipe on new data a refresh,
    while a recipe that scores worse on the same folds is refused.
    """
    g = cfg["gate"]
    if not np.isfinite(candidate.r) or candidate.r < float(g["min_r"]):
        return GateDecision(False, f"candidate R {candidate.r:.3f} is below the floor {g['min_r']}")
    if incumbent is None:
        return GateDecision(True, "no incumbent; candidate clears the R floor")
    tol = float(g["rmse_tolerance_mgL"])
    if candidate.rmse_mgL > incumbent.rmse_mgL + tol:
        return GateDecision(False, f"RMSE {candidate.rmse_mgL:.0f} > incumbent "
                                   f"{incumbent.rmse_mgL:.0f} + {tol:.0f} mg/L")
    if candidate.r < incumbent.r - float(g["max_r_drop"]):
        return GateDecision(False, f"R {candidate.r:.3f} < incumbent {incumbent.r:.3f} "
                                   f"- {g['max_r_drop']}")
    return GateDecision(True, f"RMSE {candidate.rmse_mgL:.0f} vs {incumbent.rmse_mgL:.0f} mg/L, "
                              f"R {candidate.r:.3f} vs {incumbent.r:.3f}")


# --- refit ----------------------------------------------------------------

def refit(cfg: Config, acidity, weather, station: str, recipe: Recipe, epochs: int) -> tuple:
    """Train on every year for a fixed `epochs`, early stopping off.

    Returns (members, scaler, sample_set). Every row is training data, so the
    scaler is fitted on all of it.
    """
    s = sample_set_for(cfg, acidity, weather, station, recipe)
    s = s.with_splits(np.array(["train"] * len(s), dtype=object))
    rcfg = override(cfg, {"train": {"max_epochs": int(epochs), "patience": int(epochs) + 1}})
    scaler = fit_scaler(s.X, s.y, s.feature_names, ddof=int(cfg["scaling"].get("ddof", 0)),
                        fit_on="all", station=station, window_type=recipe.window_type)
    members, results, _ = train_recipe(s, rcfg, recipe, scaler=scaler)
    assert all(r.epochs_run == int(epochs) for r in results), "refit stopped early"
    return members, scaler, s
