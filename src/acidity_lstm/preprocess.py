"""Cleaning: calendar reindexing, flag handling, acidity coercion, duplicates.

Missing weather is never imputed (CLAUDE.md rule 8). Gaps stay NaN and the
affected samples are dropped later, when windows are built.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .io import read_acidity_raw, read_weather_raw

log = logging.getLogger(__name__)

WEATHER_COLUMNS = ["date", "precip_mm", "tmean_c"]
ACIDITY_COLUMNS = ["station", "date", "acidity_mgL"]


@dataclass
class CleaningReport:
    """Everything dropped or altered during cleaning, for the audit report."""

    weather_days: int = 0
    weather_precip_nan: int = 0
    weather_tmean_nan: int = 0
    weather_flag_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    acidity_rows_in: int = 0
    acidity_non_numeric: int = 0
    acidity_out_of_range: int = 0
    acidity_duplicate_dates: int = 0
    acidity_rows_out: dict[str, int] = field(default_factory=dict)
    non_numeric_rows: pd.DataFrame | None = None
    duplicate_rows: pd.DataFrame | None = None


def clean_weather(cfg: Config, raw: pd.DataFrame | None = None) -> tuple[pd.DataFrame, CleaningReport]:
    """Reindex weather onto a continuous daily calendar; keep gaps as NaN.

    Environment Canada flags are applied as configured: "M" (missing) forces
    NaN and "T" (trace) forces 0 mm. In this dataset every T row already
    carries 0.0, so the trace rule is a no-op safeguard.
    """
    rep = CleaningReport()
    w = read_weather_raw(cfg) if raw is None else raw.copy()

    data = cfg["data"]
    rep.weather_flag_counts = {
        "precip": w["precip_flag"].replace("", "(none)").value_counts().to_dict(),
        "tmean": w["tmean_flag"].replace("", "(none)").value_counts().to_dict(),
    }

    if data.get("missing_flag_as_nan", True):
        w.loc[w["precip_flag"].eq("M"), "precip_mm"] = np.nan
        w.loc[w["tmean_flag"].eq("M"), "tmean_c"] = np.nan
    if data.get("trace_precip_as_zero", True):
        w.loc[w["precip_flag"].eq("T"), "precip_mm"] = 0.0

    if w["date"].duplicated().any():
        dupes = int(w["date"].duplicated().sum())
        log.warning("Weather has %d duplicate dates; keeping the first of each.", dupes)
        w = w.drop_duplicates(subset="date", keep="first")

    calendar = pd.date_range(data["weather_start"], data["weather_end"], freq="D")
    w = (
        w.set_index("date")
        .reindex(calendar)          # gaps stay NaN; never imputed
        .rename_axis("date")
        .reset_index()[WEATHER_COLUMNS]
    )

    rep.weather_days = len(w)
    rep.weather_precip_nan = int(w["precip_mm"].isna().sum())
    rep.weather_tmean_nan = int(w["tmean_c"].isna().sum())
    return w, rep


def clean_acidity(cfg: Config, raw: pd.DataFrame | None = None) -> tuple[pd.DataFrame, CleaningReport]:
    """Coerce acidity to numeric, clip to the study period, resolve duplicates."""
    rep = CleaningReport()
    a = read_acidity_raw(cfg) if raw is None else raw.copy()
    rep.acidity_rows_in = len(a)

    data = cfg["data"]
    a["acidity_mgL"] = pd.to_numeric(a["acidity_mgL"], errors="coerce")

    bad = a["acidity_mgL"].isna()
    rep.acidity_non_numeric = int(bad.sum())
    rep.non_numeric_rows = a.loc[bad, ["station", "date", "sheet", "excel_row"]].copy()
    if rep.acidity_non_numeric:
        log.info("Dropping %d acidity rows with no numeric value.", rep.acidity_non_numeric)
    a = a.loc[~bad].copy()

    start = pd.Timestamp(data["acidity_start"])
    end = pd.Timestamp(data["acidity_end"])
    in_range = a["date"].between(start, end)
    rep.acidity_out_of_range = int((~in_range).sum())
    a = a.loc[in_range].copy()

    dup_mask = a.duplicated(subset=["station", "date"], keep=False)
    rep.acidity_duplicate_dates = int(dup_mask.sum())
    rep.duplicate_rows = a.loc[dup_mask, ["station", "date", "acidity_mgL"]].sort_values(
        ["station", "date"]
    )

    policy = data.get("same_day_duplicates", "mean")
    if policy == "mean" and rep.acidity_duplicate_dates:
        log.info("Averaging %d same-day duplicate acidity rows.", rep.acidity_duplicate_dates)
        a = (
            a.groupby(["station", "date"], as_index=False)["acidity_mgL"]
            .mean()
        )
    elif policy not in ("mean", "keep"):
        raise ValueError(f"Unknown same_day_duplicates policy: {policy!r}")

    a = a[ACIDITY_COLUMNS].sort_values(["station", "date"]).reset_index(drop=True)
    rep.acidity_rows_out = a.groupby("station").size().to_dict()
    return a, rep


def gap_runs(dates: pd.Series, missing: pd.Series) -> pd.DataFrame:
    """Contiguous runs of missing values as `start, end, n_days`.

    `dates` must be a continuous daily calendar aligned with `missing`.
    """
    missing = np.asarray(missing, dtype=bool)
    dates = pd.to_datetime(pd.Series(dates).reset_index(drop=True))
    if not missing.any():
        return pd.DataFrame(columns=["start", "end", "n_days"])

    # A run starts wherever a True is not preceded by a True.
    starts = np.flatnonzero(missing & ~np.r_[False, missing[:-1]])
    ends = np.flatnonzero(missing & ~np.r_[missing[1:], False])
    return pd.DataFrame(
        {
            "start": dates.iloc[starts].to_numpy(),
            "end": dates.iloc[ends].to_numpy(),
            "n_days": ends - starts + 1,
        }
    )


def build_processed(cfg: Config, save: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, CleaningReport, CleaningReport]:
    """Run both cleaners and optionally write parquet to `processed_dir`."""
    weather, wrep = clean_weather(cfg)
    acidity, arep = clean_acidity(cfg)

    if save:
        out = Path(cfg.resolve("processed_dir"))
        out.mkdir(parents=True, exist_ok=True)
        weather.to_parquet(out / "weather_daily.parquet", index=False)
        acidity.to_parquet(out / "acidity_long.parquet", index=False)
        log.info("Wrote processed parquet to %s", out)

    return weather, acidity, wrep, arep
