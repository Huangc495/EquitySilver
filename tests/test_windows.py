"""Window-construction tests required by CLAUDE.md Phase 2."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from acidity_lstm.config import load_config
from acidity_lstm.windows import (
    build_sample_sets,
    restrict_to_common,
    sample_counts,
    DAYS_PER_WEEK,
    _day_index_matrix,
    build_samples,
    make_weather_arrays,
)

WEATHER_START = "1997-01-01"
WEATHER_END = "1998-12-31"


@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture
def weather():
    """A synthetic daily calendar with no gaps.

    precip = day offset, tmean = -day offset, so a window's contents identify
    the exact days it covers.
    """
    dates = pd.date_range(WEATHER_START, WEATHER_END, freq="D")
    n = len(dates)
    return pd.DataFrame(
        {
            "date": dates,
            "precip_mm": np.arange(n, dtype=float),
            "tmean_c": -np.arange(n, dtype=float),
        }
    )


def acidity_on(dates, station="BD", value=10000.0):
    return pd.DataFrame(
        {
            "station": station,
            "date": pd.to_datetime(list(dates)),
            "acidity_mgL": float(value),
        }
    )


def covered_days(arrays, anchor_date, n_steps, window_type):
    """The calendar days a window covers, oldest first."""
    anchor = arrays.offsets([pd.Timestamp(anchor_date)])
    idx = _day_index_matrix(anchor, n_steps, window_type)[0]
    origin = arrays.origin
    return origin + pd.to_timedelta(np.asarray(idx).ravel(), unit="D")


# --- 1. Window boundaries -------------------------------------------------

def test_type_a_window_boundaries(weather):
    arrays = make_weather_arrays(weather)
    days = covered_days(arrays, "1998-03-15", 10, "A")
    assert days[0] == pd.Timestamp("1998-03-06")
    assert days[-1] == pd.Timestamp("1998-03-15")
    assert len(days) == 10


def test_type_b_window_boundaries(weather):
    arrays = make_weather_arrays(weather)
    idx = _day_index_matrix(arrays.offsets([pd.Timestamp("1998-03-15")]), 10, "B")[0]
    origin = arrays.origin

    assert idx.shape == (10, DAYS_PER_WEEK)
    # Steps are oldest-first, so week 0 of the paper is the LAST step.
    week0 = origin + pd.to_timedelta(idx[-1], unit="D")
    week9 = origin + pd.to_timedelta(idx[0], unit="D")

    assert week0[0] == pd.Timestamp("1998-03-09")
    assert week0[-1] == pd.Timestamp("1998-03-15")
    assert week9[0] == pd.Timestamp("1998-01-05")
    assert week9[-1] == pd.Timestamp("1998-01-11")

    all_days = origin + pd.to_timedelta(idx.ravel(), unit="D")
    assert len(set(all_days)) == 70


# --- 2. 1997 lookback -----------------------------------------------------

def test_type_b_reaches_into_1997(weather):
    arrays = make_weather_arrays(weather)
    days = covered_days(arrays, "1998-01-20", 10, "B")
    assert days.min() == pd.Timestamp("1997-11-12")
    assert days.max() == pd.Timestamp("1998-01-20")


# --- 3. Missing-day handling ----------------------------------------------

def test_nan_30_days_back_drops_type_b_but_keeps_type_a(cfg, weather):
    anchor = pd.Timestamp("1998-06-15")
    gap_day = anchor - pd.Timedelta(days=30)
    w = weather.copy()
    w.loc[w["date"] == gap_day, "precip_mm"] = np.nan

    acidity = acidity_on([anchor])
    a = build_samples(acidity, w, cfg, "BD", "A")
    b = build_samples(acidity, w, cfg, "BD", "B")

    assert len(a) == 1, "a gap 30 days back is outside the 10-day Type A window"
    assert len(b) == 0, "a gap 30 days back falls inside the 70-day Type B window"
    assert b.n_dropped == 1


def test_window_outside_weather_record_is_dropped(cfg, weather):
    # 1997-01-05 needs days back to 1996-12-27, which the record does not have.
    acidity = acidity_on([pd.Timestamp("1997-01-05")])
    assert len(build_samples(acidity, weather, cfg, "BD", "A")) == 0


# --- 4. Aggregation -------------------------------------------------------

def test_type_b_aggregation_matches_hand_calculation(cfg):
    dates = pd.date_range("1997-01-01", "1998-12-31", freq="D")
    rng = np.random.default_rng(0)
    precip = rng.uniform(0, 20, len(dates))
    tmean = rng.uniform(-25, 25, len(dates))
    w = pd.DataFrame({"date": dates, "precip_mm": precip, "tmean_c": tmean})

    anchor = pd.Timestamp("1998-03-15")
    s = build_samples(acidity_on([anchor]), w, cfg, "BD", "B")
    assert len(s) == 1

    by_date = dict(zip(dates, zip(precip, tmean)))
    for k in range(10):
        j = 9 - k                                  # step k is week j, oldest first
        last = anchor - pd.Timedelta(days=7 * j)
        week = [last - pd.Timedelta(days=i) for i in range(DAYS_PER_WEEK)]
        expected_p = sum(by_date[d][0] for d in week)
        expected_t = sum(by_date[d][1] for d in week) / DAYS_PER_WEEK
        assert s.X[0, k, 0] == pytest.approx(expected_p, rel=1e-5)
        assert s.X[0, k, 1] == pytest.approx(expected_t, rel=1e-5)


def test_type_a_values_match_the_daily_series(cfg, weather):
    anchor = pd.Timestamp("1998-03-15")
    s = build_samples(acidity_on([anchor]), weather, cfg, "BD", "A")
    offset = (anchor - pd.Timestamp(WEATHER_START)).days
    expected = np.arange(offset - 9, offset + 1, dtype=float)
    assert np.allclose(s.X[0, :, 0], expected)
    assert np.allclose(s.X[0, :, 1], -expected)


# --- 5. Ordering ----------------------------------------------------------

def test_steps_run_oldest_to_newest(cfg, weather):
    anchor = pd.Timestamp("1998-06-15")
    a = build_samples(acidity_on([anchor]), weather, cfg, "BD", "A")
    # precip encodes the day offset, so it must increase along the step axis.
    assert np.all(np.diff(a.X[0, :, 0]) > 0)

    b = build_samples(acidity_on([anchor]), weather, cfg, "BD", "B")
    assert np.all(np.diff(b.X[0, :, 0]) > 0)

    last_offset = (anchor - pd.Timestamp(WEATHER_START)).days
    assert a.X[0, -1, 0] == pytest.approx(last_offset)


def test_tensor_layout(cfg, weather):
    dates = pd.date_range("1998-04-01", periods=12, freq="14D")
    s = build_samples(acidity_on(dates), weather, cfg, "BD", "B")
    assert s.X.shape == (len(s), 10, 2)
    assert s.y.shape == (len(s),)
    assert len(s.meta) == len(s)
    assert list(s.meta["date"]) == sorted(s.meta["date"])


# --- 6. Time tag ----------------------------------------------------------

def test_time_tag_per_step_type_a(cfg, weather):
    anchor = pd.Timestamp("1998-03-15")
    expected_anchor_day = (anchor - pd.Timestamp("1998-01-01")).days + 1
    assert expected_anchor_day == 74

    s = build_samples(acidity_on([anchor]), weather, cfg, "BD", "A", with_time_tag=True)
    assert s.X.shape == (1, 10, 3)
    assert s.feature_names == ("precip", "tmean", "day_number")
    assert np.allclose(s.X[0, :, 2], np.arange(65, 75, dtype=float))


def test_time_tag_per_step_type_b_uses_week_last_day(cfg, weather):
    anchor = pd.Timestamp("1998-03-15")
    s = build_samples(acidity_on([anchor]), weather, cfg, "BD", "B", with_time_tag=True)
    # Week j ends on d - 7j, so day numbers step by 7 and end at 74.
    assert np.allclose(s.X[0, :, 2], np.arange(74 - 63, 75, 7, dtype=float))
    assert s.X[0, -1, 2] == pytest.approx(74.0)


def test_time_tag_constant(cfg, weather):
    import copy

    c = copy.deepcopy(cfg.raw)
    c["windows"]["time_tag"] = "constant"
    cfg_const = type(cfg)(raw=c, paths=cfg.paths, repo_root=cfg.repo_root)

    anchor = pd.Timestamp("1998-03-15")
    s = build_samples(acidity_on([anchor]), weather, cfg_const, "BD", "B", with_time_tag=True)
    assert np.allclose(s.X[0, :, 2], 74.0)


def test_day_number_origin_is_one(cfg, weather):
    s = build_samples(acidity_on([pd.Timestamp("1998-01-15")]), weather, cfg, "BD", "A")
    assert int(s.meta["day_number"].iloc[0]) == 15


# --- 7. Sample-set assembly and the common-sample-set option --------------

def with_windows(cfg, **updates):
    """A copy of `cfg` with the `windows` section shallow-updated."""
    import copy

    raw = copy.deepcopy(cfg.raw)
    raw["windows"].update(updates)
    return type(cfg)(raw=raw, paths=cfg.paths, repo_root=cfg.repo_root)


def test_build_sample_sets_keys_and_shapes(cfg, weather):
    dates = pd.date_range("1998-04-01", periods=10, freq="14D")
    acidity = pd.concat([acidity_on(dates, "BD"), acidity_on(dates, "C7")])

    sets = build_sample_sets(acidity, weather, cfg, types=["A", "B"])
    assert set(sets) == {("BD", "A"), ("BD", "B"), ("C7", "A"), ("C7", "B")}
    for (station, wt), s in sets.items():
        assert s.station == station and s.window_type == wt
        assert s.X.shape[1:] == (10, 2)


# Anchors more than 70 days apart, so no two Type B windows overlap and a
# single gap day can be attributed to exactly one sample.
SPACED_ANCHORS = pd.to_datetime(["1998-04-01", "1998-08-01", "1998-12-01"])
# 30 days before the middle anchor: inside its Type B window, inside no
# Type A window, and inside no other sample's window.
LONE_GAP_DAY = SPACED_ANCHORS[1] - pd.Timedelta(days=30)


def test_common_sample_set_restricts_type_a_to_type_b_survivors(cfg, weather):
    anchors = SPACED_ANCHORS
    w = weather.copy()
    w.loc[w["date"] == LONE_GAP_DAY, "tmean_c"] = np.nan
    acidity = acidity_on(anchors)

    default = build_sample_sets(acidity, w, with_windows(cfg, common_sample_set=False),
                               stations=["BD"])
    assert len(default[("BD", "A")]) == 3
    assert len(default[("BD", "B")]) == 2

    common = build_sample_sets(acidity, w, with_windows(cfg, common_sample_set=True),
                              stations=["BD"])
    assert len(common[("BD", "A")]) == 2, "Type A must drop to the Type B survivors"
    assert len(common[("BD", "B")]) == 2
    assert set(common[("BD", "A")].meta["date"]) == set(common[("BD", "B")].meta["date"])


def test_restrict_to_common_keeps_rows_aligned(cfg, weather):
    anchors = SPACED_ANCHORS
    w = weather.copy()
    w.loc[w["date"] == LONE_GAP_DAY, "tmean_c"] = np.nan
    acidity = acidity_on(anchors)

    sets = {wt: build_samples(acidity, w, cfg, "BD", wt) for wt in ("A", "B")}
    common = restrict_to_common(sets)
    for s in common.values():
        assert len(s.X) == len(s.y) == len(s.meta)
        assert np.allclose(s.y, s.meta["acidity_mgL"].to_numpy())


def test_sample_counts_table(cfg, weather):
    dates = pd.date_range("1998-04-01", periods=8, freq="14D")
    sets = build_sample_sets(acidity_on(dates), weather, cfg, stations=["BD"])
    tab = sample_counts(sets)
    assert list(tab["type"]) == ["A", "B"]
    assert (tab["surviving"] + tab["dropped"] == tab["candidates"]).all()


# --- 8. Real-data survival counts (locks the Phase 0 / Phase 2 result) ----

@pytest.fixture(scope="module")
def real_data():
    from acidity_lstm.preprocess import clean_acidity, clean_weather

    c = load_config()
    weather, _ = clean_weather(c)
    acidity, _ = clean_acidity(c)
    return c, weather, acidity


@pytest.mark.integration
def test_real_survival_counts(real_data):
    cfg_r, weather, acidity = real_data
    sets = build_sample_sets(acidity, weather, cfg_r)
    got = {(st, wt): len(s) for (st, wt), s in sets.items()}
    assert got == {
        ("BD", "A"): 274,
        ("BD", "B"): 185,
        ("C7", "A"): 278,
        ("C7", "B"): 186,
    }


@pytest.mark.integration
def test_real_type_b_is_a_subset_of_type_a(real_data):
    cfg_r, weather, acidity = real_data
    sets = build_sample_sets(acidity, weather, cfg_r)
    for station in ("BD", "C7"):
        a = set(sets[(station, "A")].meta["date"])
        b = set(sets[(station, "B")].meta["date"])
        assert b <= a, f"{station}: Type B survivors must all survive Type A"


@pytest.mark.integration
def test_real_samples_have_no_nan_features(real_data):
    cfg_r, weather, acidity = real_data
    for tag in (False, True):
        sets = build_sample_sets(acidity, weather, cfg_r, with_time_tag=tag)
        for key, s in sets.items():
            assert not np.isnan(s.X).any(), f"{key}: NaN leaked into the inputs"
            assert not np.isnan(s.y).any()
            assert s.X.dtype == np.float32


@pytest.mark.integration
def test_real_time_tag_adds_a_third_feature(real_data):
    cfg_r, weather, acidity = real_data
    plain = build_sample_sets(acidity, weather, cfg_r, stations=["BD"], types=["B"])
    tagged = build_sample_sets(acidity, weather, cfg_r, stations=["BD"], types=["B"],
                               with_time_tag=True)
    p, t = plain[("BD", "B")], tagged[("BD", "B")]
    assert p.X.shape[2] == 2 and t.X.shape[2] == 3
    # The time tag must not change which samples survive.
    assert len(p) == len(t)
    assert np.allclose(p.X, t.X[:, :, :2])
    # Day numbers must be strictly increasing over the study period.
    assert (np.diff(t.meta["day_number"].to_numpy()) > 0).all()
