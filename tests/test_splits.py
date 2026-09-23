"""Split-assignment tests (CLAUDE.md Phase 5)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from acidity_lstm.config import Config, load_config
from acidity_lstm.splits import (
    SPLIT_NAMES,
    _largest_remainder,
    assign_splits,
    blocked_split,
    eligible_years_from,
    random_split,
    realised_counts,
)


@pytest.fixture
def cfg():
    return load_config()


def with_split(cfg: Config, **updates) -> Config:
    import copy

    raw = copy.deepcopy(cfg.raw)
    raw["split"].update(updates)
    return Config(raw=raw, paths=cfg.paths, repo_root=cfg.repo_root)


def fake_acidity(station="BD", start="1998-01-01", periods=200, freq="14D"):
    return pd.DataFrame(
        {
            "station": station,
            "date": pd.date_range(start, periods=periods, freq=freq),
            "acidity_mgL": 12000.0,
        }
    )


# --- Size allocation ------------------------------------------------------

def test_largest_remainder_sums_to_n():
    for n in (1, 7, 100, 365, 384):
        sizes = _largest_remainder(n, [0.70, 0.15, 0.15])
        assert sum(sizes) == n


def test_random_split_respects_the_70_15_15_fractions():
    labels = random_split(200, [0.70, 0.15, 0.15], seed=0)
    counts = pd.Series(labels).value_counts()
    assert counts["train"] == 140
    assert counts["val"] == 30
    assert counts["test"] == 30


def test_fractions_must_sum_to_one(cfg):
    with pytest.raises(ValueError, match="sum to 1"):
        random_split(100, [0.7, 0.2, 0.2], seed=0)


def test_wrong_number_of_fractions_raises():
    with pytest.raises(ValueError, match="fractions"):
        random_split(100, [0.8, 0.2], seed=0)


# --- Determinism ----------------------------------------------------------

def test_same_seed_gives_the_same_split(cfg):
    a = assign_splits(fake_acidity(), cfg, seed=42, station="BD")
    b = assign_splits(fake_acidity(), cfg, seed=42, station="BD")
    assert a.equals(b)


def test_different_seeds_give_different_splits(cfg):
    a = assign_splits(fake_acidity(), cfg, seed=1, station="BD")
    b = assign_splits(fake_acidity(), cfg, seed=2, station="BD")
    assert not a.equals(b)


def test_split_is_indexed_by_date_and_covers_every_measurement(cfg):
    acidity = fake_acidity(periods=150)
    s = assign_splits(acidity, cfg, seed=0, station="BD")

    assert isinstance(s.index, pd.DatetimeIndex)
    assert len(s) == 150
    assert set(s.index) == set(acidity["date"])
    assert set(s.unique()) <= set(SPLIT_NAMES)


def test_split_maps_onto_samples_by_date(cfg):
    """The mechanism `windows.build_samples(splits=...)` relies on."""
    acidity = fake_acidity(periods=50)
    s = assign_splits(acidity, cfg, seed=0, station="BD")
    mapped = acidity["date"].map(s)
    assert mapped.notna().all()
    assert (mapped.to_numpy() == s.loc[acidity["date"]].to_numpy()).all()


def test_unknown_method_raises(cfg):
    with pytest.raises(ValueError, match="split.method"):
        assign_splits(fake_acidity(), with_split(cfg, method="kfold"), seed=0, station="BD")


def test_empty_station_raises(cfg):
    with pytest.raises(ValueError, match="No measurements"):
        assign_splits(fake_acidity(), cfg, seed=0, station="XX")


# --- Blocked splits -------------------------------------------------------

def test_blocked_split_keeps_whole_years_together(cfg):
    acidity = fake_acidity(periods=400)          # ~15 years
    s = assign_splits(acidity, with_split(cfg, method="blocked"), seed=0, station="BD")

    by_year = pd.DataFrame({"year": s.index.year, "split": s.to_numpy()})
    per_year = by_year.groupby("year")["split"].nunique()
    assert (per_year == 1).all(), "a year must not be split across folds"


def test_blocked_split_fills_every_fold(cfg):
    acidity = fake_acidity(periods=400)
    s = assign_splits(acidity, with_split(cfg, method="blocked"), seed=3, station="BD")
    assert set(s.unique()) == set(SPLIT_NAMES)


def test_blocked_split_test_years_are_disjoint_from_train(cfg):
    acidity = fake_acidity(periods=400)
    s = assign_splits(acidity, with_split(cfg, method="blocked"), seed=1, station="BD")
    years = {name: set(s.index[s == name].year) for name in SPLIT_NAMES}
    assert not (years["train"] & years["test"])
    assert not (years["train"] & years["val"])
    assert not (years["val"] & years["test"])


def test_blocked_split_restricted_to_eligible_years():
    """Q-03: years emptied by the complete-window rule must not form a fold."""
    dates = pd.Series(pd.date_range("1998-01-01", periods=500, freq="14D"))
    eligible = [1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2008, 2009]

    labels = blocked_split(dates, [0.70, 0.15, 0.15], seed=0, eligible_years=eligible)
    assigned = pd.DataFrame({"year": dates.dt.year, "split": labels})

    # Ineligible years must never land in val or test.
    bad = assigned[(~assigned["year"].isin(eligible)) & (assigned["split"] != "train")]
    assert bad.empty
    # Every fold still gets real years.
    held_out = set(assigned[assigned["split"].isin(["val", "test"])]["year"])
    assert held_out and held_out <= set(eligible)


def test_blocked_split_needs_enough_years():
    dates = pd.Series(pd.date_range("2000-01-01", periods=30, freq="14D"))
    with pytest.raises(ValueError, match="at least"):
        blocked_split(dates, [0.70, 0.15, 0.15], seed=0, eligible_years=[2000, 2001])


def test_blocked_split_with_no_eligible_years_raises():
    dates = pd.Series(pd.date_range("2000-01-01", periods=100, freq="14D"))
    with pytest.raises(ValueError, match="No eligible years"):
        blocked_split(dates, [0.70, 0.15, 0.15], seed=0, eligible_years=[1950])


def test_small_eligible_pool_still_fills_all_folds():
    """10 eligible years at 70/15/15 rounds to 7/2/1 -- no empty fold."""
    dates = pd.Series(pd.date_range("1999-01-01", periods=400, freq="14D"))
    eligible = list(range(1999, 2009))
    labels = blocked_split(dates, [0.70, 0.15, 0.15], seed=0, eligible_years=eligible)
    held = pd.DataFrame({"year": dates.dt.year, "split": labels})
    held = held[held["year"].isin(eligible)]
    assert set(held["split"].unique()) == set(SPLIT_NAMES)


# --- Integration with real sample sets ------------------------------------

@pytest.fixture(scope="module")
def real_sets():
    from acidity_lstm.preprocess import clean_acidity, clean_weather
    from acidity_lstm.windows import build_sample_sets

    c = load_config()
    weather, _ = clean_weather(c)
    acidity, _ = clean_acidity(c)
    return c, acidity, build_sample_sets(acidity, weather, c)


@pytest.mark.integration
def test_eligible_years_excludes_the_empty_years(real_sets):
    _, _, sets = real_sets
    years = eligible_years_from({k: v for k, v in sets.items() if k[1] == "B"})
    for empty in (1998, 2007, 2010, 2011, 2012, 2017):
        assert empty not in years, f"{empty} has no Type B survivors"
    assert len(years) >= 3


@pytest.mark.integration
def test_split_is_shared_across_input_types(real_sets):
    """One split per station, inherited by every input type (Phase 5)."""
    cfg_r, acidity, _ = real_sets
    from acidity_lstm.windows import build_sample_sets

    from acidity_lstm.preprocess import clean_weather

    weather, _ = clean_weather(cfg_r)
    splits = {
        st: assign_splits(acidity, cfg_r, seed=42, station=st)
        for st in ("BD", "C7")
    }
    sets = build_sample_sets(acidity, weather, cfg_r, splits_by_station=splits)

    a = sets[("BD", "A")].meta.set_index("date")["split"]
    b = sets[("BD", "B")].meta.set_index("date")["split"]
    shared = a.index.intersection(b.index)
    assert len(shared) > 0
    assert (a.loc[shared] == b.loc[shared]).all(), "types must agree on shared dates"


@pytest.mark.integration
def test_realised_proportions_drift_from_70_15_15(real_sets):
    """Expected: dropping is uneven, so realised splits are not exactly 70/15/15."""
    cfg_r, acidity, _ = real_sets
    from acidity_lstm.preprocess import clean_weather
    from acidity_lstm.windows import build_sample_sets

    weather, _ = clean_weather(cfg_r)
    splits = {"BD": assign_splits(acidity, cfg_r, seed=42, station="BD")}
    sets = build_sample_sets(acidity, weather, cfg_r, stations=["BD"],
                             splits_by_station=splits)
    counts = realised_counts(sets[("BD", "B")])

    assert sum(counts.values()) == len(sets[("BD", "B")]) == 185
    assert all(v > 0 for v in counts.values()), "every fold must keep some samples"
    train_frac = counts["train"] / sum(counts.values())
    assert 0.55 < train_frac < 0.85, f"train fraction {train_frac:.2f} looks wrong"


@pytest.mark.integration
def test_no_sample_is_left_unassigned(real_sets):
    cfg_r, acidity, _ = real_sets
    from acidity_lstm.preprocess import clean_weather
    from acidity_lstm.windows import build_sample_sets

    weather, _ = clean_weather(cfg_r)
    splits = {st: assign_splits(acidity, cfg_r, seed=7, station=st) for st in ("BD", "C7")}
    sets = build_sample_sets(acidity, weather, cfg_r, splits_by_station=splits)
    for key, s in sets.items():
        assert s.meta["split"].isin(SPLIT_NAMES).all(), f"{key} has unassigned rows"
