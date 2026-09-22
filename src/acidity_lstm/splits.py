"""Train / validation / test assignment.

Splits are assigned **once per station, before windowing**, on the acidity
measurements themselves. Samples inherit the assignment of their measurement
date, so every input type and hidden size for a station shares one split
(CLAUDE.md Phase 5).

Because the complete-window rule drops samples unevenly, the realised split
proportions after windowing will not be exactly 70/15/15. That is a
consequence of sharing one split across input types, and is reported rather
than corrected.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .config import Config

log = logging.getLogger(__name__)

SPLIT_NAMES = ("train", "val", "test")


def assign_splits(
    acidity: pd.DataFrame,
    cfg: Config,
    seed: int,
    station: str | None = None,
    eligible_years=None,
) -> pd.Series:
    """Assign each measurement of one station to train / val / test.

    Returns a Series of split labels indexed by measurement date, ready to be
    passed to `windows.build_samples(splits=...)`.

    `method: random` draws per measurement; `method: blocked` holds out whole
    calendar years. `eligible_years` restricts the blocked draw to years that
    actually contain usable samples (see the note in `blocked_split`).
    """
    scfg = cfg["split"]
    method = scfg["method"]
    fractions = list(scfg["fractions"])

    sub = acidity if station is None else acidity.loc[acidity["station"] == station]
    dates = pd.to_datetime(sub["date"]).sort_values().reset_index(drop=True)
    if dates.empty:
        raise ValueError(f"No measurements to split for station {station!r}.")

    if method == "random":
        labels = random_split(len(dates), fractions, seed)
    elif method == "blocked":
        labels = blocked_split(dates, fractions, seed, eligible_years=eligible_years)
    else:
        raise ValueError(f"Unknown split.method {method!r}; expected 'random' or 'blocked'.")

    out = pd.Series(labels, index=pd.DatetimeIndex(dates), name="split")
    counts = out.value_counts().reindex(SPLIT_NAMES).fillna(0).astype(int)
    log.info(
        "station=%s method=%s seed=%d -> train %d / val %d / test %d measurements",
        station, method, seed, counts["train"], counts["val"], counts["test"],
    )
    return out


def random_split(n: int, fractions, seed: int) -> np.ndarray:
    """A seeded per-measurement 70/15/15 draw (the paper's hold-out approach).

    Sizes are allocated by largest remainder so they always sum to `n`.
    """
    _check_fractions(fractions)
    sizes = _largest_remainder(n, fractions)
    labels = np.concatenate([np.repeat(name, k) for name, k in zip(SPLIT_NAMES, sizes)])
    np.random.default_rng(seed).shuffle(labels)
    return labels


def blocked_split(dates: pd.Series, fractions, seed: int, eligible_years=None) -> np.ndarray:
    """Hold out whole calendar years (robustness check, not the paper).

    Samples are roughly two weeks apart, so consecutive Type B windows share
    about 55 of their 70 days. A random split therefore puts near-identical
    inputs in both training and test, flattering the scores. Holding out whole
    years removes that leak.

    `eligible_years` exists because the complete-window rule empties some years
    entirely: BD Type B has no surviving sample in 1998, 2007, 2010, 2011, 2012
    or 2017. Drawing folds blindly from all 20 years could put only empty years
    in the test fold. When `eligible_years` is given, only those years are
    partitioned; the remaining years are labelled `train`, where they
    contribute nothing because they hold no surviving samples.
    """
    _check_fractions(fractions)
    dates = pd.to_datetime(pd.Series(dates).reset_index(drop=True))
    years = dates.dt.year.to_numpy()

    all_years = np.array(sorted(set(years.tolist())))
    if eligible_years is None:
        pool = all_years
    else:
        pool = np.array(sorted(set(all_years.tolist()) & set(int(y) for y in eligible_years)))
        if len(pool) == 0:
            raise ValueError("No eligible years left to build a blocked split from.")

    if len(pool) < len(SPLIT_NAMES):
        raise ValueError(
            f"Only {len(pool)} eligible years; need at least {len(SPLIT_NAMES)} "
            "to fill every fold."
        )

    shuffled = pool.copy()
    np.random.default_rng(seed).shuffle(shuffled)
    sizes = _largest_remainder(len(shuffled), fractions)
    # Guarantee a non-empty fold even when the pool is small.
    sizes = _ensure_nonempty(sizes, len(shuffled))

    year_to_split: dict[int, str] = {}
    start = 0
    for name, k in zip(SPLIT_NAMES, sizes):
        for y in shuffled[start:start + k]:
            year_to_split[int(y)] = name
        start += k

    excluded = sorted(set(all_years.tolist()) - set(year_to_split))
    if excluded:
        log.info(
            "Blocked split: years %s hold no usable samples and are labelled 'train'.",
            excluded,
        )

    return np.array([year_to_split.get(int(y), "train") for y in years], dtype=object)


def eligible_years_from(sample_sets) -> list[int]:
    """Years that contain at least one surviving sample in every given set.

    Pass the most restrictive scenarios (typically Type B) so the blocked split
    is drawn from years usable by all of them.
    """
    sets = sample_sets.values() if isinstance(sample_sets, dict) else list(sample_sets)
    common = None
    for s in sets:
        years = set(s.meta["date"].dt.year.astype(int))
        common = years if common is None else (common & years)
    return sorted(common or [])


def realised_counts(sample_set) -> dict:
    """Split sizes actually realised after windowing, for logging."""
    counts = sample_set.meta["split"].value_counts()
    return {name: int(counts.get(name, 0)) for name in SPLIT_NAMES}


def _check_fractions(fractions) -> None:
    if len(fractions) != len(SPLIT_NAMES):
        raise ValueError(f"Expected {len(SPLIT_NAMES)} fractions, got {len(fractions)}.")
    if any(f < 0 for f in fractions):
        raise ValueError("Split fractions must be non-negative.")
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError(f"Split fractions must sum to 1, got {sum(fractions)}.")


def _largest_remainder(n: int, fractions) -> list[int]:
    """Integer sizes summing to `n`, allocated by largest remainder."""
    exact = [n * f for f in fractions]
    sizes = [int(np.floor(e)) for e in exact]
    remainder = n - sum(sizes)
    order = np.argsort([-(e - np.floor(e)) for e in exact])
    for i in range(remainder):
        sizes[order[i % len(sizes)]] += 1
    return sizes


def _ensure_nonempty(sizes: list[int], total: int) -> list[int]:
    """Move one unit from the largest fold into any empty fold."""
    sizes = list(sizes)
    if total < len(sizes):
        return sizes
    for i, k in enumerate(sizes):
        if k == 0:
            donor = int(np.argmax(sizes))
            if sizes[donor] > 1:
                sizes[donor] -= 1
                sizes[i] += 1
    return sizes
