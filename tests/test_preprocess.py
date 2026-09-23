"""Phase 1 cleaning tests.

These lock the invariants the audit established, above all that cleaning
reproduces the paper's own sample counts (BD 365, C7 384).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from acidity_lstm.config import Config, load_config
from acidity_lstm.preprocess import clean_acidity, clean_weather, gap_runs

PAPER_COUNTS = {"BD": 365, "C7": 384}


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def cleaned(cfg):
    weather, _ = clean_weather(cfg)
    acidity, arep = clean_acidity(cfg)
    return weather, acidity, arep


def with_overrides(cfg: Config, **section_updates) -> Config:
    """A copy of `cfg` with nested sections shallow-updated."""
    import copy

    raw = copy.deepcopy(cfg.raw)
    for section, updates in section_updates.items():
        raw[section].update(updates)
    return Config(raw=raw, paths=cfg.paths, repo_root=cfg.repo_root)


# --- Weather --------------------------------------------------------------

@pytest.mark.integration
def test_weather_is_a_continuous_daily_calendar(cfg, cleaned):
    weather, _, _ = cleaned
    expected = pd.date_range(
        cfg["data"]["weather_start"], cfg["data"]["weather_end"], freq="D"
    )
    assert len(weather) == len(expected)
    assert weather["date"].reset_index(drop=True).equals(pd.Series(expected))
    assert not weather["date"].duplicated().any()
    assert list(weather.columns) == ["date", "precip_mm", "tmean_c"]


@pytest.mark.integration
def test_weather_gaps_are_never_imputed(cleaned):
    weather, _, _ = cleaned
    # The audit found 781 / 1038 missing values; cleaning must not fill them.
    assert weather["precip_mm"].isna().sum() > 0
    assert weather["tmean_c"].isna().sum() > 0
    assert weather["precip_mm"].isna().sum() == 781
    assert weather["tmean_c"].isna().sum() == 1038


@pytest.mark.integration
def test_weather_values_are_physically_plausible(cleaned):
    weather, _, _ = cleaned
    p = weather["precip_mm"].dropna()
    t = weather["tmean_c"].dropna()
    assert (p >= 0).all(), "negative precipitation"
    assert t.between(-60, 45).all(), "temperature outside a plausible range"


def test_missing_flag_forces_nan_and_trace_forces_zero(cfg):
    raw = pd.DataFrame(
        {
            "date": pd.date_range("1997-01-01", periods=4, freq="D"),
            "precip_mm": [5.0, 9.9, 0.0, 1.0],
            "tmean_c": [1.0, 2.0, 7.7, 4.0],
            "precip_flag": ["", "M", "T", ""],
            "tmean_flag": ["", "", "M", ""],
        }
    )
    small = with_overrides(cfg, data={"weather_start": "1997-01-01",
                                      "weather_end": "1997-01-04"})
    out, _ = clean_weather(small, raw=raw)

    assert np.isnan(out.loc[1, "precip_mm"]), "M must force NaN"
    assert out.loc[2, "precip_mm"] == 0.0, "T must force 0 mm"
    assert np.isnan(out.loc[2, "tmean_c"]), "M must force NaN"
    assert out.loc[0, "precip_mm"] == 5.0, "unflagged values are untouched"


def test_reindex_inserts_nan_rows_for_absent_dates(cfg):
    raw = pd.DataFrame(
        {
            "date": pd.to_datetime(["1997-01-01", "1997-01-04"]),
            "precip_mm": [1.0, 2.0],
            "tmean_c": [3.0, 4.0],
            "precip_flag": ["", ""],
            "tmean_flag": ["", ""],
        }
    )
    small = with_overrides(cfg, data={"weather_start": "1997-01-01",
                                      "weather_end": "1997-01-04"})
    out, _ = clean_weather(small, raw=raw)
    assert len(out) == 4
    assert out["precip_mm"].isna().tolist() == [False, True, True, False]


# --- Acidity --------------------------------------------------------------

@pytest.mark.integration
def test_acidity_counts_match_the_paper(cleaned):
    """The headline Phase 0 finding, locked as a regression test."""
    _, acidity, _ = cleaned
    assert acidity.groupby("station").size().to_dict() == PAPER_COUNTS


@pytest.mark.integration
def test_blank_acidity_cells_are_dropped(cleaned):
    _, acidity, arep = cleaned
    assert arep.acidity_non_numeric == 5
    assert acidity["acidity_mgL"].notna().all()


@pytest.mark.integration
def test_acidity_is_clipped_to_the_study_period(cfg, cleaned):
    _, acidity, _ = cleaned
    start = pd.Timestamp(cfg["data"]["acidity_start"])
    end = pd.Timestamp(cfg["data"]["acidity_end"])
    assert acidity["date"].between(start, end).all()
    # C7 reaches back to 1986 in the raw file, so plenty must be dropped.
    assert acidity["date"].min() >= start


@pytest.mark.integration
def test_acidity_has_no_same_day_duplicates(cleaned):
    _, acidity, arep = cleaned
    assert arep.acidity_duplicate_dates == 0
    assert not acidity.duplicated(["station", "date"]).any()


@pytest.mark.integration
def test_acidity_is_sorted_and_positive(cleaned):
    _, acidity, _ = cleaned
    assert acidity.equals(
        acidity.sort_values(["station", "date"]).reset_index(drop=True)
    )
    assert (acidity["acidity_mgL"] > 0).all()
    assert list(acidity.columns) == ["station", "date", "acidity_mgL"]


def test_same_day_duplicates_are_averaged_when_they_occur(cfg):
    raw = pd.DataFrame(
        {
            "station": ["BD", "BD", "BD"],
            "date": pd.to_datetime(["2000-05-01", "2000-05-01", "2000-05-02"]),
            "acidity_mgL": [10000.0, 20000.0, 12000.0],
            "sheet": "x",
            "excel_row": [4, 5, 6],
        }
    )
    out, rep = clean_acidity(cfg, raw=raw)
    assert rep.acidity_duplicate_dates == 2
    assert len(out) == 2
    assert out.loc[out["date"] == pd.Timestamp("2000-05-01"), "acidity_mgL"].iloc[0] == 15000.0


def test_keep_policy_retains_duplicates(cfg):
    raw = pd.DataFrame(
        {
            "station": ["BD", "BD"],
            "date": pd.to_datetime(["2000-05-01", "2000-05-01"]),
            "acidity_mgL": [10000.0, 20000.0],
            "sheet": "x",
            "excel_row": [4, 5],
        }
    )
    keep_cfg = with_overrides(cfg, data={"same_day_duplicates": "keep"})
    out, _ = clean_acidity(keep_cfg, raw=raw)
    assert len(out) == 2


def test_unknown_duplicate_policy_raises(cfg):
    raw = pd.DataFrame(
        {
            "station": ["BD", "BD"],
            "date": pd.to_datetime(["2000-05-01", "2000-05-01"]),
            "acidity_mgL": [1.0, 2.0],
            "sheet": "x",
            "excel_row": [4, 5],
        }
    )
    bad = with_overrides(cfg, data={"same_day_duplicates": "median"})
    with pytest.raises(ValueError, match="same_day_duplicates"):
        clean_acidity(bad, raw=raw)


# --- gap_runs -------------------------------------------------------------

def test_gap_runs_identifies_contiguous_runs():
    dates = pd.date_range("2000-01-01", periods=10, freq="D")
    missing = [False, True, True, False, False, True, False, False, False, True]
    runs = gap_runs(pd.Series(dates), pd.Series(missing))

    assert len(runs) == 3
    assert runs["n_days"].tolist() == [2, 1, 1]
    assert pd.Timestamp(runs.iloc[0]["start"]) == pd.Timestamp("2000-01-02")
    assert pd.Timestamp(runs.iloc[0]["end"]) == pd.Timestamp("2000-01-03")
    # A run that reaches the end of the record must still be closed.
    assert pd.Timestamp(runs.iloc[-1]["end"]) == pd.Timestamp("2000-01-10")


def test_gap_runs_with_no_missing_values():
    dates = pd.date_range("2000-01-01", periods=5, freq="D")
    runs = gap_runs(pd.Series(dates), pd.Series([False] * 5))
    assert len(runs) == 0


@pytest.mark.integration
def test_gap_runs_total_matches_missing_count(cleaned):
    weather, _, _ = cleaned
    either = weather["precip_mm"].isna() | weather["tmean_c"].isna()
    runs = gap_runs(weather["date"], either)
    assert runs["n_days"].sum() == int(either.sum())
