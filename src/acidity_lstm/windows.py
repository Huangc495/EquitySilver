"""Type A / Type B lookback windows and the optional time tag (paper Fig. 5).

Every window is anchored on the sample's own date d and includes day d itself,
so the steps run from -9 to 0. Weather lives in NumPy arrays indexed by the
integer day offset from `weather_start`; windows are cut by vectorised index
arithmetic rather than per-sample pandas filtering.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import Config

log = logging.getLogger(__name__)

DAYS_PER_WEEK = 7
FEATURES = ("precip", "tmean")


@dataclass
class WeatherArrays:
    """Daily weather as offset-indexed arrays.

    `precip[i]` and `tmean[i]` hold the values for `origin + i` days.
    """

    origin: pd.Timestamp
    precip: np.ndarray
    tmean: np.ndarray

    def __len__(self) -> int:
        return len(self.precip)

    def offsets(self, dates) -> np.ndarray:
        """Integer day offsets of `dates` from the origin."""
        d = pd.to_datetime(pd.Series(np.asarray(dates)).reset_index(drop=True))
        return (d - self.origin).dt.days.to_numpy()


@dataclass
class SampleSet:
    """Model inputs and the row-aligned metadata used for figures."""

    X: np.ndarray               # (N, n_steps, n_features)
    y: np.ndarray               # (N,) acidity in mg/L
    meta: pd.DataFrame          # station, date, acidity_mgL, day_number, split
    station: str
    window_type: str            # "A" or "B"
    feature_names: tuple
    n_candidates: int           # samples before the complete-window rule
    n_dropped: int

    def __len__(self) -> int:
        return len(self.y)

    def select(self, mask) -> "SampleSet":
        """A new SampleSet holding only the rows where `mask` is true."""
        mask = np.asarray(mask, dtype=bool)
        return SampleSet(
            X=self.X[mask],
            y=self.y[mask],
            meta=self.meta.loc[mask].reset_index(drop=True),
            station=self.station,
            window_type=self.window_type,
            feature_names=self.feature_names,
            n_candidates=self.n_candidates,
            n_dropped=self.n_candidates - int(mask.sum()),
        )

    def with_splits(self, splits) -> "SampleSet":
        """A copy whose `split` column is replaced by `splits`."""
        meta = self.meta.copy()
        meta["split"] = np.asarray(splits, dtype=object)
        return SampleSet(
            X=self.X, y=self.y, meta=meta,
            station=self.station, window_type=self.window_type,
            feature_names=self.feature_names,
            n_candidates=self.n_candidates, n_dropped=self.n_dropped,
        )


def make_weather_arrays(weather: pd.DataFrame) -> WeatherArrays:
    """Convert the cleaned daily weather frame into offset-indexed arrays.

    The frame must already be a continuous daily calendar (see
    `preprocess.clean_weather`).
    """
    w = weather.sort_values("date").reset_index(drop=True)
    dates = pd.to_datetime(w["date"]).reset_index(drop=True)
    expected = pd.Series(pd.date_range(dates.iloc[0], dates.iloc[-1], freq="D"))
    if len(w) != len(expected) or not dates.equals(expected):
        raise ValueError("Weather frame is not a continuous daily calendar.")

    return WeatherArrays(
        origin=dates.iloc[0],
        precip=w["precip_mm"].to_numpy(dtype=float),
        tmean=w["tmean_c"].to_numpy(dtype=float),
    )


def _day_index_matrix(anchor: np.ndarray, n_steps: int, window_type: str) -> np.ndarray:
    """Day offsets covered by each sample's window.

    Returns (N, n_steps) for Type A and (N, n_steps, 7) for Type B, ordered
    oldest to newest along the step axis.

    Type A step k (k = 0..n_steps-1) is day d - (n_steps-1-k).
    Type B week j (j = 0..n_steps-1, counted back from the sample) spans
    [d-7j-6, d-7j]; step k corresponds to j = n_steps-1-k.
    """
    if window_type == "A":
        lag = np.arange(n_steps - 1, -1, -1)                  # oldest -> newest
        return anchor[:, None] - lag[None, :]

    if window_type == "B":
        j = np.arange(n_steps - 1, -1, -1)[None, :, None]     # oldest week first
        within = np.arange(DAYS_PER_WEEK - 1, -1, -1)[None, None, :]
        return anchor[:, None, None] - DAYS_PER_WEEK * j - within

    raise ValueError("Unknown window type " + repr(window_type) + "; expected 'A' or 'B'.")


def _time_tag(
    n_steps: int,
    window_type: str,
    anchor_day_number: np.ndarray,
    mode: str,
) -> np.ndarray:
    """Day-number feature of shape (N, n_steps).

    `per_step`: each step carries its own day number -- the step's day for
    Type A, the week's last day (d - 7j) for Type B.
    `constant`: every step carries the sample's own day number.
    """
    if mode == "constant":
        return np.repeat(anchor_day_number[:, None], n_steps, axis=1).astype(float)

    if mode != "per_step":
        raise ValueError("Unknown time_tag mode " + repr(mode) + "; expected 'per_step' or 'constant'.")

    step = 1 if window_type == "A" else DAYS_PER_WEEK
    lag = step * np.arange(n_steps - 1, -1, -1)
    return (anchor_day_number[:, None] - lag[None, :]).astype(float)


def build_samples(
    acidity: pd.DataFrame,
    weather,
    cfg: Config,
    station: str,
    window_type: str,
    with_time_tag: bool = False,
    splits=None,
) -> SampleSet:
    """Build the (N, n_steps, n_features) tensor for one station and window type.

    A sample is dropped if any day in its window is NaN in either variable, or
    if the window reaches outside the weather record. Nothing is imputed.
    """
    wcfg = cfg["windows"]
    n_steps = int(wcfg["n_steps"])
    arrays = weather if isinstance(weather, WeatherArrays) else make_weather_arrays(weather)

    sub = (
        acidity.loc[acidity["station"] == station]
        .sort_values("date")
        .reset_index(drop=True)
        .copy()
    )
    n_candidates = len(sub)
    if n_candidates == 0:
        raise ValueError("No acidity samples for station " + repr(station) + ".")

    anchor = arrays.offsets(sub["date"])
    idx = _day_index_matrix(anchor, n_steps, window_type)

    # Windows reaching outside the weather record are invalid, not imputable.
    in_range = (idx >= 0) & (idx < len(arrays))
    safe = np.where(in_range, idx, 0)

    precip = arrays.precip[safe]
    tmean = arrays.tmean[safe]
    valid = in_range & ~np.isnan(precip) & ~np.isnan(tmean)

    if window_type == "A":
        feat = np.stack([precip, tmean], axis=-1)              # (N, n_steps, 2)
        keep = valid.all(axis=1)
    else:
        # Weekly precipitation is a sum; weekly temperature is a mean.
        feat = np.stack([precip.sum(axis=2), tmean.mean(axis=2)], axis=-1)
        keep = valid.all(axis=(1, 2))

    feature_names = list(FEATURES)
    origin = pd.Timestamp(wcfg["time_tag_origin"])
    day_number = ((pd.to_datetime(sub["date"]) - origin).dt.days + 1).to_numpy()

    if with_time_tag:
        tag = _time_tag(n_steps, window_type, day_number, wcfg["time_tag"])
        feat = np.concatenate([feat, tag[:, :, None]], axis=-1)
        feature_names.append("day_number")

    meta = pd.DataFrame(
        {
            "station": station,
            "date": pd.to_datetime(sub["date"]),
            "acidity_mgL": sub["acidity_mgL"].to_numpy(dtype=float),
            "day_number": day_number,
        }
    )
    if splits is not None:
        meta["split"] = meta["date"].map(splits).to_numpy()
    else:
        meta["split"] = pd.NA

    out = SampleSet(
        X=feat[keep].astype(np.float32),
        y=meta.loc[keep, "acidity_mgL"].to_numpy(dtype=np.float32),
        meta=meta.loc[keep].reset_index(drop=True),
        station=station,
        window_type=window_type,
        feature_names=tuple(feature_names),
        n_candidates=n_candidates,
        n_dropped=int((~keep).sum()),
    )
    log.info(
        "station=%s type=%s: %d/%d samples survive the complete-window rule (%d dropped).",
        station, window_type, len(out), n_candidates, out.n_dropped,
    )
    return out


def build_sample_sets(
    acidity: pd.DataFrame,
    weather,
    cfg: Config,
    stations=None,
    types=None,
    with_time_tag: bool = False,
    splits_by_station: dict | None = None,
) -> dict:
    """Build every (station, window type) SampleSet the config asks for.

    Applies `windows.common_sample_set` per station, so that when it is true
    Type A is restricted to the samples that also survive Type B. Returns a
    dict keyed by `(station, window_type)`.
    """
    wcfg = cfg["windows"]
    stations = list(stations if stations is not None else cfg["data"]["stations"])
    types = list(types if types is not None else wcfg["types"])
    arrays = weather if isinstance(weather, WeatherArrays) else make_weather_arrays(weather)

    out: dict = {}
    for station in stations:
        splits = (splits_by_station or {}).get(station)
        per_station = {
            wt: build_samples(
                acidity, arrays, cfg, station, wt,
                with_time_tag=with_time_tag, splits=splits,
            )
            for wt in types
        }
        if wcfg.get("common_sample_set", False) and len(per_station) > 1:
            per_station = restrict_to_common(per_station)
            log.info(
                "station=%s: common sample set applied, %d samples per type.",
                station, len(next(iter(per_station.values()))),
            )
        for wt, s in per_station.items():
            out[(station, wt)] = s
    return out


def sample_counts(sets: dict) -> pd.DataFrame:
    """Summarise a dict of SampleSets as a counts table."""
    rows = [
        {
            "station": s.station,
            "type": s.window_type,
            "n_features": s.X.shape[2],
            "candidates": s.n_candidates,
            "surviving": len(s),
            "dropped": s.n_dropped,
            "pct_kept": round(100 * len(s) / s.n_candidates, 1),
            "first": s.meta["date"].min().strftime("%Y-%m-%d") if len(s) else "",
            "last": s.meta["date"].max().strftime("%Y-%m-%d") if len(s) else "",
        }
        for s in sets.values()
    ]
    return pd.DataFrame(rows).sort_values(["station", "type"]).reset_index(drop=True)


def restrict_to_common(sets: dict) -> dict:
    """Restrict several SampleSets for one station to their shared dates.

    Used by `windows.common_sample_set: true`, so Type A and Type B are
    compared on identical samples.
    """
    common = None
    for s in sets.values():
        dates = set(s.meta["date"])
        common = dates if common is None else (common & dates)

    out = {}
    for name, s in sets.items():
        keep = s.meta["date"].isin(common).to_numpy()
        out[name] = SampleSet(
            X=s.X[keep],
            y=s.y[keep],
            meta=s.meta.loc[keep].reset_index(drop=True),
            station=s.station,
            window_type=s.window_type,
            feature_names=s.feature_names,
            n_candidates=s.n_candidates,
            n_dropped=s.n_candidates - int(keep.sum()),
        )
    return out
