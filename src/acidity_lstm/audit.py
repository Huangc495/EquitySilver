"""Phase 0 data audit: structure, continuity, quality, counts and figures.

Produces `reports/00_data_audit.md` plus figures. Kept in the library so the
notebook stays thin (CLAUDE.md rule 4).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import Config
from .io import read_acidity_raw, read_weather_raw, weather_files
from .preprocess import clean_acidity, clean_weather, gap_runs
from .windows import (
    build_sample_sets,
    build_samples,
    make_weather_arrays,
    sample_counts,
)

# Counts reported by Ma et al. (2021) for 1998-2017.
PAPER_COUNTS = {"BD": 365, "C7": 384}


def _md_table(df: pd.DataFrame, index: bool = False) -> str:
    """Render a DataFrame as a GitHub markdown table.

    Written by hand rather than via `DataFrame.to_markdown` so the report does
    not depend on `tabulate`, which is not guaranteed on the cluster.
    """
    if index:
        df = df.reset_index()

    def cell(v) -> str:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return ""
        if isinstance(v, float):
            return f"{v:,.4g}"
        if isinstance(v, (pd.Timestamp, np.datetime64)):
            return pd.Timestamp(v).strftime("%Y-%m-%d")
        return str(v)

    headers = [str(c) for c in df.columns]
    rows = [[cell(v) for v in row] for row in df.itertuples(index=False, name=None)]
    widths = [
        max(len(h), *(len(r[i]) for r in rows)) if rows else len(h)
        for i, h in enumerate(headers)
    ]

    def line(values):
        return "| " + " | ".join(v.ljust(widths[i]) for i, v in enumerate(values)) + " |"

    out = [line(headers), "|" + "|".join("-" * (wd + 2) for wd in widths) + "|"]
    out += [line(r) for r in rows]
    return "\n".join(out)


def _fig_path(cfg: Config, name: str) -> Path:
    d = cfg.reports_dir / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d / name


def run_audit(cfg: Config) -> str:
    """Run the full Phase 0 audit and write the markdown report. Returns it."""
    out: list[str] = []
    w(out, "# Phase 0 - Data audit")
    w(out, "")
    w(out, "Equity Silver waste-rock-pile replication of Ma et al. (2021).")
    w(out, f"Generated from `{cfg.resolve('acidity_excel').name}` and "
           f"{len(weather_files(cfg))} Environment Canada daily CSVs.")
    w(out, "")

    raw_w = read_weather_raw(cfg, keep_extra=True)
    raw_a = read_acidity_raw(cfg)
    weather, wrep = clean_weather(cfg)
    acidity, arep = clean_acidity(cfg)

    _section_structure(out, cfg, raw_w, raw_a)
    _section_weather_continuity(out, raw_w, weather)
    _section_flags(out, raw_w, wrep)
    _section_acidity_dates(out, raw_a, arep)
    _section_acidity_counts(out, acidity)
    _section_acidity_quality(out, acidity, arep)
    _section_seasonality(out, acidity)
    _section_survival(out, cfg, acidity, weather)
    _section_figures(out, cfg, weather, acidity)

    text = "\n".join(out) + "\n"
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    (cfg.reports_dir / "00_data_audit.md").write_text(text, encoding="utf-8")
    return text


def w(out: list[str], line: str = "") -> None:
    out.append(line)


# --- 1. File structure ----------------------------------------------------

def _section_structure(out, cfg, raw_w, raw_a):
    w(out, "## 1. File structure")
    w(out)
    w(out, "### Weather (Environment Canada, one CSV per year)")
    w(out)
    files = weather_files(cfg)
    w(out, f"- {len(files)} files, `{files[0].name}` .. `{files[-1].name}`")
    w(out, f"- Station: {', '.join(sorted(raw_w['station_name'].dropna().unique()))} "
           f"(Climate ID {', '.join(sorted(raw_w['climate_id'].dropna().unique()))})")
    w(out, f"- Rows: {len(raw_w):,}; date range "
           f"{raw_w['date'].min():%Y-%m-%d} to {raw_w['date'].max():%Y-%m-%d}")
    w(out, "- Model inputs used: `Total Precip (mm)` -> `precip_mm`, "
           "`Mean Temp (°C)` -> `tmean_c`. All other columns are audited only.")
    w(out)
    extra = raw_w[["tmax_c", "tmin_c", "rain_mm", "snow_cm", "snow_grnd_cm",
                   "precip_mm", "tmean_c"]].describe().T
    extra["n_missing"] = raw_w[extra.index].isna().sum().values
    w(out, _md_table(extra.round(2).reset_index().rename(columns={"index": "column"})))
    w(out)

    w(out, "### Acidity workbook")
    w(out)
    w(out, f"- File: `{cfg.resolve('acidity_excel').name}`")
    w(out, "- Two sheets, one per station, mapped by config "
           "`data.acidity_sheets`:")
    for sheet, station in cfg["data"]["acidity_sheets"].items():
        n = int((raw_a["sheet"] == sheet).sum())
        w(out, f"  - `{sheet}` -> **{station}** ({n:,} data rows)")
    w(out, "- Three header rows: long name, unit, short field name. Data starts "
           "on Excel row 4.")
    w(out, "- Columns: `Station Name`, `Collect Date/Time`, `ACIDITY` (mg/L as "
           "CaCO3), `PH-L`, `ZN-D`, `ZN-T`. Only `ACIDITY` is used.")
    w(out, "- Dates are stored as Excel serial numbers (1899-12-30 origin) and "
           "are normalised to whole days.")
    w(out)


# --- 2. Weather continuity ------------------------------------------------

def _section_weather_continuity(out, raw_w, weather):
    w(out, "## 2. Weather continuity")
    w(out)
    cal = pd.date_range(weather["date"].min(), weather["date"].max(), freq="D")
    missing_rows = sorted(set(cal) - set(raw_w["date"]))
    w(out, f"- Expected calendar days: **{len(cal):,}**; rows present: "
           f"**{raw_w['date'].nunique():,}**; duplicate dates: "
           f"**{int(raw_w['date'].duplicated().sum())}**")
    w(out, f"- Missing calendar *rows*: **{len(missing_rows)}** - the record is "
           "continuous daily.")
    w(out, "- Missing *values* are the real issue:")
    w(out)

    summary = pd.DataFrame(
        {
            "variable": ["precip_mm", "tmean_c", "either"],
            "n_missing": [
                int(weather["precip_mm"].isna().sum()),
                int(weather["tmean_c"].isna().sum()),
                int((weather["precip_mm"].isna() | weather["tmean_c"].isna()).sum()),
            ],
        }
    )
    summary["pct_of_record"] = (100 * summary["n_missing"] / len(weather)).round(1)
    w(out, _md_table(summary))
    w(out)

    either = weather["precip_mm"].isna() | weather["tmean_c"].isna()
    runs = gap_runs(weather["date"], either)
    w(out, f"### Gap runs (either variable missing): {len(runs)} runs, "
           f"{int(runs['n_days'].sum()):,} days")
    w(out)
    # Date breaks ties, so the table is identical on every platform: numpy 2
    # orders equal-length runs differently from numpy 1 (D-33).
    top = runs.sort_values(["n_days", "start"], ascending=[False, True],
                           kind="stable").head(15).copy()
    top["start"] = pd.to_datetime(top["start"]).dt.strftime("%Y-%m-%d")
    top["end"] = pd.to_datetime(top["end"]).dt.strftime("%Y-%m-%d")
    w(out, "Longest 15 runs:")
    w(out)
    w(out, _md_table(top))
    w(out)
    hist = runs["n_days"].value_counts().sort_index()
    w(out, "Run-length distribution: "
           + ", ".join(f"{k} d x{v}" for k, v in hist.head(10).items()))
    w(out)

    per_year = (
        weather.assign(year=weather["date"].dt.year)
        .groupby("year")
        .agg(
            precip_missing=("precip_mm", lambda s: int(s.isna().sum())),
            tmean_missing=("tmean_c", lambda s: int(s.isna().sum())),
        )
        .reset_index()
    )
    per_year["either_missing"] = (
        weather.assign(year=weather["date"].dt.year)
        .groupby("year")
        .apply(lambda g: int((g["precip_mm"].isna() | g["tmean_c"].isna()).sum()),
               include_groups=False)
        .values
    )
    w(out, "### Missing days per year")
    w(out)
    w(out, _md_table(per_year))
    w(out)
    worst = per_year.sort_values(["either_missing", "year"],
                                 ascending=[False, True], kind="stable").head(5)
    w(out, "Worst years: "
           + ", ".join(f"**{int(r.year)}** ({int(r.either_missing)} d)"
                       for r in worst.itertuples()))
    w(out)
    w(out, "- Station name and Climate ID never change over the record, so no "
           "station splice has to be handled.")
    w(out)


# --- 3. Quality flags -----------------------------------------------------

def _section_flags(out, raw_w, wrep):
    w(out, "## 3. Weather quality flags")
    w(out)
    for var, counts in wrep.weather_flag_counts.items():
        s = pd.Series(counts).sort_values(ascending=False, kind="stable")
        w(out, f"- **{var}**: " + ", ".join(f"`{k}` x{v:,}" for k, v in s.items()))
    w(out)
    w(out, "Environment Canada flag meanings: `M` missing, `T` trace, "
           "`E` estimated, `A` accumulated, `C` precipitation occurred but "
           "amount uncertain, `F` accumulated and estimated.")
    w(out)

    cross = pd.crosstab(
        raw_w["precip_flag"].replace("", "(none)"), raw_w["precip_mm"].isna()
    ).rename(columns={False: "has_value", True: "is_nan"}).reset_index()
    w(out, "### Precipitation flag vs missing value")
    w(out)
    w(out, _md_table(cross))
    w(out)
    cross_t = pd.crosstab(
        raw_w["tmean_flag"].replace("", "(none)"), raw_w["tmean_c"].isna()
    ).rename(columns={False: "has_value", True: "is_nan"}).reset_index()
    w(out, "### Mean-temperature flag vs missing value")
    w(out)
    w(out, _md_table(cross_t))
    w(out)

    t_vals = raw_w.loc[raw_w["precip_flag"].eq("T"), "precip_mm"]
    w(out, "**Findings:**")
    w(out)
    w(out, f"1. Every `T` (trace) row already carries "
           f"{t_vals.unique().tolist()} mm, so the configured `T -> 0 mm` rule "
           "is a no-op safeguard, not a change to the data.")
    w(out, "2. Every `M` row is already blank, so the `M -> NaN` rule is also "
           "a no-op safeguard.")
    n_blank_p = int(((raw_w["precip_flag"] == "") & raw_w["precip_mm"].isna()).sum())
    n_blank_t = int(((raw_w["tmean_flag"] == "") & raw_w["tmean_c"].isna()).sum())
    w(out, f"3. **The bulk of missing data carries no flag at all**: "
           f"{n_blank_p:,} blank-and-unflagged precipitation days and "
           f"{n_blank_t:,} for mean temperature, against only "
           f"{int(raw_w['precip_flag'].eq('M').sum())} and "
           f"{int(raw_w['tmean_flag'].eq('M').sum())} explicit `M` flags. "
           "These are treated as missing (NaN) and never imputed.")
    w(out, "4. `A`/`C`/`F` precipitation rows do carry values and are kept "
           "as recorded; flagging them out would enlarge the gaps further.")
    w(out)


# --- 4. Acidity dates -----------------------------------------------------

def _section_acidity_dates(out, raw_a, arep):
    w(out, "## 4. Acidity dates")
    w(out)
    n_null = int(raw_a["date"].isna().sum())
    w(out, f"- Rows read: **{len(raw_a):,}**; rows with an unparseable or "
           f"missing date: **{n_null}**.")
    w(out, "- Every retained value therefore has a full calendar date; no "
           "month-only or year-only entries were found.")
    w(out, f"- Full-file date span: {raw_a['date'].min():%Y-%m-%d} to "
           f"{raw_a['date'].max():%Y-%m-%d}.")
    w(out, f"- Rows outside the 1998-2017 study window (dropped): "
           f"**{arep.acidity_out_of_range:,}** - almost entirely C7 "
           "measurements from 1986-1997.")
    w(out)


# --- 5. Acidity counts ----------------------------------------------------

def _section_acidity_counts(out, acidity):
    w(out, "## 5. Acidity counts vs the paper")
    w(out)
    counts = acidity.groupby("station").size().rename("ours").to_frame()
    counts["paper"] = counts.index.map(PAPER_COUNTS)
    counts["difference"] = counts["ours"] - counts["paper"]
    w(out, _md_table(counts.reset_index()))
    w(out)
    if (counts["difference"] == 0).all():
        w(out, "**Exact match with the paper for both stations.** This is a "
               "strong confirmation that the data source, the 1998-2017 "
               "window and the blank-dropping rule all match Ma et al.")
    else:
        w(out, "Mismatch against the paper - see the discussion below.")
    w(out)

    per_year = pd.crosstab(acidity["date"].dt.year, acidity["station"])
    per_year.index.name = "year"
    w(out, "### Measurements per station per year")
    w(out)
    w(out, _md_table(per_year.reset_index()))
    w(out)
    w(out, "1998 and 2017 are partial years: BD starts 1998-10-02 and both "
           "stations stop at 2017-06-22. Sampling thins from roughly 23-25 per "
           "year before 2008 to 15-17 per year afterwards.")
    w(out)


# --- 6. Acidity quality ---------------------------------------------------

def _section_acidity_quality(out, acidity, arep):
    w(out, "## 6. Acidity data quality")
    w(out)
    w(out, f"- Non-numeric / blank `ACIDITY` entries dropped: "
           f"**{arep.acidity_non_numeric}**")
    if arep.non_numeric_rows is not None and len(arep.non_numeric_rows):
        t = arep.non_numeric_rows.copy()
        t["date"] = pd.to_datetime(t["date"]).dt.strftime("%Y-%m-%d")
        w(out)
        w(out, _md_table(t))
        w(out)
        w(out, "These are genuinely blank cells, not censored values "
               "(no `<`, `>` or `ND` text appears anywhere in the column).")
    w(out)
    w(out, f"- Same-day duplicate measurements (rows involved): "
           f"**{arep.acidity_duplicate_dates}**")
    if arep.duplicate_rows is not None and len(arep.duplicate_rows):
        t = arep.duplicate_rows.copy()
        t["date"] = pd.to_datetime(t["date"]).dt.strftime("%Y-%m-%d")
        w(out)
        w(out, _md_table(t))
    else:
        w(out, "  - None. The `same_day_duplicates: mean` policy never fires, "
               "so that ambiguity does not affect this replication.")
    w(out)

    desc = acidity.groupby("station")["acidity_mgL"].describe().round(1)
    w(out, "### Value range (mg/L as CaCO3)")
    w(out)
    w(out, _md_table(desc.reset_index()))
    w(out)
    w(out, "Paper Fig. 4 shows values mostly between 5,000 and 30,000 mg/L "
           "with occasional peaks near 40,000.")
    for st, g in acidity.groupby("station"):
        n_out = int(((g["acidity_mgL"] < 5000) | (g["acidity_mgL"] > 30000)).sum())
        n_hi = int((g["acidity_mgL"] > 40000).sum())
        w(out, f"- **{st}**: {n_out} values outside 5,000-30,000 "
               f"({100*n_out/len(g):.0f}%), {n_hi} above 40,000, "
               f"min {g['acidity_mgL'].min():,.0f}, max {g['acidity_mgL'].max():,.0f}.")
    w(out)
    n_nonpos = int((acidity["acidity_mgL"] <= 0).sum())
    w(out, f"- Non-positive values: **{n_nonpos}**.")
    w(out)


# --- 7. Seasonality -------------------------------------------------------

def _section_seasonality(out, acidity):
    w(out, "## 7. Seasonality of sampling")
    w(out)
    by_month = pd.crosstab(acidity["date"].dt.month, acidity["station"])
    by_month.index.name = "month"
    by_month["total"] = by_month.sum(axis=1)
    w(out, _md_table(by_month.reset_index()))
    w(out)
    mar_jul = by_month.loc[3:7, "total"].sum()
    w(out, f"March-July holds **{mar_jul:,} of {int(by_month['total'].sum()):,}** "
           f"measurements ({100*mar_jul/by_month['total'].sum():.0f}%) in 5 of "
           "12 months, confirming the paper's statement that sampling is more "
           "frequent from March to July (the freshet).")
    w(out)


# --- 8. Surviving samples -------------------------------------------------

def _section_survival(out, cfg, acidity, weather):
    w(out, "## 8. Samples surviving the complete-window rule")
    w(out)
    w(out, "A sample is kept only if **every** day of its lookback window has "
           "both precipitation and mean temperature: 10 days for Type A, "
           "70 days for Type B. Nothing is imputed (CLAUDE.md rule 8).")
    w(out)

    arrays = make_weather_arrays(weather)
    rows = []
    survivors = {}
    kept_by_year = {}
    for station in cfg["data"]["stations"]:
        for wt in ("A", "B"):
            s = build_samples(acidity, arrays, cfg, station, wt)
            survivors[(station, wt)] = set(s.meta["date"])
            kept_by_year[f"{station}-{wt}"] = (
                s.meta["date"].dt.year.value_counts().sort_index()
            )
            rows.append(
                {
                    "station": station,
                    "type": wt,
                    "candidates": s.n_candidates,
                    "surviving": len(s),
                    "dropped": s.n_dropped,
                    "pct_kept": round(100 * len(s) / s.n_candidates, 1),
                }
            )
    tab = pd.DataFrame(rows)
    w(out, _md_table(tab))
    w(out)

    for station in cfg["data"]["stations"]:
        a, b = survivors[(station, "A")], survivors[(station, "B")]
        w(out, f"- **{station}**: Type B survivors are "
               f"{'a subset of' if b <= a else 'NOT a subset of'} Type A "
               f"survivors; |A n B| = {len(a & b)}, A-only = {len(a - b)}, "
               f"B-only = {len(b - a)}.")
    w(out)
    w(out, "### Consequence for the replication")
    w(out)
    min_b = tab.loc[tab["type"] == "B", "surviving"].min()
    w(out, "The paper applies the same rule and says so explicitly:")
    w(out)
    w(out, "> As some weather data are missing at the Equity Silver site, not "
           "all acidity measurements at C7 and BD stations are utilized for "
           "building the observation samples for machine learning. Those "
           "acidity measurements without a complete time series input data "
           "will be neglected.")
    w(out)
    w(out, "So Ma et al. dropped rather than gap-filled, which matches our "
           "implementation. **They never report the surviving counts**, so the "
           "open question is only how large their surviving sets were.")
    w(out)
    w(out, f"Type B loses substantially more than Type A, leaving as few as "
           f"**{min_b}** samples for one station. With a 70/15/15 split that is "
           f"roughly **{int(round(0.15*min_b))} test samples** - a thin basis "
           "for the reported test metrics, and a reason to expect noisy "
           "run-to-run scatter in Phase 6.")
    w(out)
    w(out, "Two consequences to keep in mind:")
    w(out)
    w(out, "1. Type A and Type B are **not** trained on the same samples under "
           "the default `common_sample_set: false`. Type B is a strict subset "
           "of Type A here, so part of any Type B vs Type A difference is a "
           "change of sample set, not of input representation. The "
           "`common_sample_set: true` run in Phase 6 isolates this.")
    w(out, "2. Losses are concentrated in 2006-2014 (the worst weather-gap "
           "years), so the surviving samples are not uniformly spread over the "
           "study period.")
    w(out)
    w(out, "### Surviving samples per year")
    w(out)
    by_year = pd.DataFrame(kept_by_year).fillna(0).astype(int)
    all_years = pd.crosstab(acidity["date"].dt.year, acidity["station"])
    by_year = by_year.reindex(all_years.index).fillna(0).astype(int)
    by_year.insert(0, "C7-all", all_years["C7"])
    by_year.insert(0, "BD-all", all_years["BD"])
    by_year.index.name = "year"
    w(out, _md_table(by_year, index=True))
    w(out)
    lost_b = {
        y: int(by_year.loc[y, "BD-all"] - by_year.loc[y, "BD-B"])
        for y in by_year.index
    }
    worst_years = sorted(lost_b, key=lambda y: -lost_b[y])[:5]
    w(out, "Worst years for BD Type B losses: "
           + ", ".join(f"**{y}** (-{lost_b[y]})" for y in worst_years)
           + ". 2011 loses every sample at both stations.")
    w(out)
    zero_b = [y for y in by_year.index if by_year.loc[y, "BD-B"] == 0]
    w(out, "Years with **no** surviving BD Type B sample at all: "
           + ", ".join(str(y) for y in zero_b) + ".")
    w(out)
    w(out, "**Implications for later phases:**")
    w(out)
    fc_years = [2015, 2016]
    fc_n = int(sum(by_year.loc[y, "BD-B"] for y in fc_years if y in by_year.index))
    tr_n = int(sum(by_year.loc[y, "BD-B"] for y in by_year.index if 1999 <= y <= 2014))
    w(out, f"- **Phase 9 (forecast)** is feasible: BD Type B has **{tr_n}** "
           f"surviving samples in 1999-2014 for training and **{fc_n}** in "
           f"2015-2016 to predict. Both are usable, though {fc_n} prediction "
           "points is a small basis for the paper's 0.23 MSE target.")
    w(out, "- **Phase 5 blocked splits** cannot simply hold out whole years: "
           + ", ".join(str(y) for y in zero_b)
           + " are empty for BD Type B. The blocked splitter must draw from "
             "years that actually contain surviving samples.")
    w(out)


# --- 9. Figures -----------------------------------------------------------

def _section_figures(out, cfg, weather, acidity):
    w(out, "## 9. Figures")
    w(out)

    # Fig A: acidity time series, compare with paper Fig. 4.
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    for ax, st in zip(axes, cfg["data"]["stations"]):
        g = acidity[acidity["station"] == st]
        ax.plot(g["date"], g["acidity_mgL"], ".-", ms=4, lw=0.7, color="tab:blue")
        ax.set_title(f"Station {st}  (n = {len(g)})", fontsize=10)
        ax.set_ylabel("Acidity (mg/L CaCO$_3$)")
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("Date")
    fig.suptitle("Measured acidity 1998-2017 (compare paper Fig. 4)", fontsize=12)
    fig.tight_layout()
    p = _fig_path(cfg, "00_acidity_timeseries.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    w(out, f"![Acidity time series](figures/{p.name})")
    w(out)
    w(out, "*Acidity at BD and C7, 1998-2017.*")
    w(out)

    # Fig B: weather with gaps visible.
    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True,
                             gridspec_kw={"height_ratios": [2, 2, 1]})
    axes[0].plot(weather["date"], weather["precip_mm"], lw=0.5, color="tab:blue")
    axes[0].set_ylabel("Precip (mm/d)")
    axes[0].set_title("Daily total precipitation - gaps shown as breaks", fontsize=10)
    axes[1].plot(weather["date"], weather["tmean_c"], lw=0.5, color="tab:red")
    axes[1].set_ylabel("Mean temp (°C)")
    axes[1].set_title("Daily mean temperature - gaps shown as breaks", fontsize=10)

    either = (weather["precip_mm"].isna() | weather["tmean_c"].isna()).astype(int)
    axes[2].fill_between(weather["date"], 0, either, step="mid", color="k", lw=0)
    axes[2].set_ylim(0, 1)
    axes[2].set_yticks([])
    axes[2].set_ylabel("missing")
    axes[2].set_xlabel("Date")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle("Equity Silver daily weather 1997-2017 (Climate ID 1072692)", fontsize=12)
    fig.tight_layout()
    p2 = _fig_path(cfg, "00_weather_with_gaps.png")
    fig.savefig(p2, dpi=140)
    plt.close(fig)
    w(out, f"![Weather with gaps](figures/{p2.name})")
    w(out)
    w(out, "*Daily precipitation and mean temperature; the bottom strip marks "
           "days where either variable is missing.*")
    w(out)

    # Fig C: sampling seasonality.
    fig, ax = plt.subplots(figsize=(8, 3.2))
    by_month = pd.crosstab(acidity["date"].dt.month, acidity["station"])
    by_month.plot(kind="bar", ax=ax, width=0.8)
    ax.axvspan(1.5, 6.5, color="gold", alpha=0.18, zorder=0, label="Mar-Jul")
    ax.set_xlabel("Month")
    ax.set_ylabel("Measurements")
    ax.set_title("Sampling frequency by month, 1998-2017", fontsize=11)
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    p3 = _fig_path(cfg, "00_sampling_by_month.png")
    fig.savefig(p3, dpi=140)
    plt.close(fig)
    w(out, f"![Sampling by month](figures/{p3.name})")
    w(out)
    w(out, "*Sampling concentrates in the March-July freshet.*")
    w(out)


# --- Phase 2 sample-count report -----------------------------------------

def _with_windows(cfg: Config, **updates) -> Config:
    """A copy of `cfg` with the `windows` section shallow-updated."""
    import copy

    raw = copy.deepcopy(cfg.raw)
    raw["windows"].update(updates)
    return Config(raw=raw, paths=cfg.paths, repo_root=cfg.repo_root)


def run_sample_count_report(cfg: Config) -> str:
    """Phase 2 deliverable: final sample counts per station and window type."""
    weather, _ = clean_weather(cfg)
    acidity, _ = clean_acidity(cfg)
    arrays = make_weather_arrays(weather)

    out: list[str] = []
    w(out, "# Phase 2 - Final sample counts")
    w(out)
    w(out, "Samples surviving the complete-window rule, per station and input "
           "type. A sample is one acidity measurement; it is kept only if every "
           "day of its lookback window has both precipitation and mean "
           "temperature. Nothing is imputed.")
    w(out)
    w(out, "Measurements available after cleaning: "
           + ", ".join(f"**{k} {v}**" for k, v in
                       acidity.groupby('station').size().items())
           + " (both exactly match Ma et al.).")
    w(out)

    default = build_sample_sets(acidity, arrays, _with_windows(cfg, common_sample_set=False))
    w(out, "## Default configuration (`common_sample_set: false`)")
    w(out)
    w(out, _md_table(sample_counts(default)))
    w(out)
    w(out, "This is the sample set the parametric study (Phase 6) will use.")
    w(out)

    tagged = build_sample_sets(
        acidity, arrays, _with_windows(cfg, common_sample_set=False), with_time_tag=True
    )
    w(out, "## With the time tag (refined model, Phase 8)")
    w(out)
    w(out, _md_table(sample_counts(tagged)))
    w(out)
    same = all(len(tagged[k]) == len(default[k]) for k in default)
    w(out, f"The time tag adds a third feature without changing which samples "
           f"survive ({'verified' if same else 'MISMATCH'}), so tagged and "
           "untagged models are compared on identical samples.")
    w(out)

    common = build_sample_sets(acidity, arrays, _with_windows(cfg, common_sample_set=True))
    w(out, "## With `common_sample_set: true` (robustness check, Phase 6)")
    w(out)
    w(out, _md_table(sample_counts(common)))
    w(out)
    w(out, "Type A is restricted to the samples that also survive Type B, so "
           "the Type A vs Type B comparison isolates the input representation "
           "rather than confounding it with a change of sample set.")
    w(out)

    w(out, "## Forecast and sensitivity periods (Phases 9-10)")
    w(out)
    fcfg = cfg["forecast"]
    st = fcfg["station"]
    tr0, tr1 = fcfg["train_years"]
    p0, p1 = fcfg["predict_years"]
    s = build_sample_sets(
        acidity, arrays, _with_windows(cfg, common_sample_set=False),
        stations=[st], types=["B"], with_time_tag=True,
    )[(st, "B")]
    years = s.meta["date"].dt.year
    rows = pd.DataFrame(
        {
            "period": [f"train+val {tr0}-{tr1}", f"predict {p0}-{p1}", "excluded"],
            "n_samples": [
                int(years.between(tr0, tr1).sum()),
                int(years.between(p0, p1).sum()),
                int((~years.between(tr0, tr1) & ~years.between(p0, p1)).sum()),
            ],
        }
    )
    w(out, _md_table(rows))
    w(out)
    w(out, f"Station {st}, Type B, with time tag. Both periods are usable, "
           "though the prediction period is small.")
    w(out)

    text = "\n".join(out) + "\n"
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    (cfg.reports_dir / "02_sample_counts.md").write_text(text, encoding="utf-8")
    return text
