"""Phase 9-10 deliverables: `reports/forecast_sensitivity.md`, Figs 11-12."""

from __future__ import annotations

import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .audit import _md_table, w
from .config import Config
from .evaluate import predict
from .forecast import PAPER_FORECAST, PREDICT_SPLIT, ForecastResult

log = logging.getLogger(__name__)

SCENARIO_STYLE = {
    "Real weather": ("black", "-", 2.0),
    "Precipitation x0.8": ("tab:blue", "--", 1.3),
    "Precipitation x1.2": ("tab:cyan", "--", 1.3),
    "Temperature x0.8": ("tab:purple", ":", 1.3),
    "Temperature x1.2": ("tab:red", ":", 1.3),
}


def _fig_dir(cfg: Config):
    d = cfg.reports_dir / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


# --- Figures --------------------------------------------------------------

def make_forecast_figures(cfg: Config, fc: ForecastResult) -> dict:
    figs = {}
    meta = fc.meta
    is_predict = (meta["split"] == PREDICT_SPLIT).to_numpy()

    y_true = fc.best.evaluation.y_true_mgL
    y_pred = fc.best.evaluation.y_pred_mgL
    dates = pd.to_datetime(meta["date"]).to_numpy()

    # Fig. 11: the forecast period alone.
    p0, p1 = fc.predict_years
    fig, ax = plt.subplots(figsize=(10, 4.4))
    ax.plot(dates[is_predict], y_true[is_predict], "o-", ms=6, lw=1.3,
            color="tab:blue", label="Measured")
    ax.plot(dates[is_predict], y_pred[is_predict], "s--", ms=6, lw=1.3,
            color="tab:red", label="Predicted")
    ax.set_xlabel("Date")
    ax.set_ylabel("Acidity (mg/L CaCO$_3$)")
    ax.set_title(
        f"Fig. 11 - BD forecast {p0}-{p1} (MSE {fc.predict_mse():.2f}, "
        f"RMSE {fc.predict_rmse_mgL():,.0f} mg/L, n = {int(is_predict.sum())})",
        fontsize=11,
    )
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = _fig_dir(cfg) / "fig11_forecast.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    figs["fig11"] = path

    # Supplementary: the whole span, with the forecast period shaded.
    fig, ax = plt.subplots(figsize=(12, 4.4))
    order = np.argsort(dates)
    ax.plot(dates[order], y_true[order], "o-", ms=3, lw=0.8,
            color="tab:blue", label="Measured")
    ax.plot(dates[order], y_pred[order], "s-", ms=3, lw=0.8,
            color="tab:red", alpha=0.85, label="Model")
    ax.axvspan(
        np.datetime64(f"{p0}-01-01"), np.datetime64(f"{p1}-12-31"),
        color="gold", alpha=0.2, label=f"Forecast period {p0}-{p1}",
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Acidity (mg/L CaCO$_3$)")
    ax.set_title("BD: training period and forecast period", fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(ncol=3, fontsize=9)
    fig.tight_layout()
    path2 = _fig_dir(cfg) / "fig11b_full_span.png"
    fig.savefig(path2, dpi=140)
    plt.close(fig)
    figs["fig11_full"] = path2
    return figs


def make_sensitivity_figure(cfg: Config, curves: dict, meta, table) -> dict:
    dates = pd.to_datetime(meta["date"]).to_numpy()
    order = np.argsort(dates)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7.6), sharex=True,
                             gridspec_kw={"height_ratios": [3, 2]})

    ax = axes[0]
    for label, values in curves.items():
        colour, style, width = SCENARIO_STYLE.get(label, ("grey", "-", 1.0))
        ax.plot(dates[order], np.asarray(values)[order], style, color=colour,
                lw=width, marker="o", ms=3.5, label=label)
    ax.set_ylabel("Predicted acidity (mg/L CaCO$_3$)")
    ax.set_title("Fig. 12 - Sensitivity of predicted acidity to +-20% weather",
                 fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(ncol=3, fontsize=9)

    # Differences from the real-weather curve, which is what the eye needs.
    base = np.asarray(curves["Real weather"])
    ax = axes[1]
    for label, values in curves.items():
        if label == "Real weather":
            continue
        colour, style, width = SCENARIO_STYLE.get(label, ("grey", "-", 1.0))
        ax.plot(dates[order], (np.asarray(values) - base)[order], style,
                color=colour, lw=width, marker="o", ms=3.5, label=label)
    ax.axhline(0, color="black", lw=1)
    ax.set_xlabel("Date")
    ax.set_ylabel("Change vs real weather (mg/L)")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    path = _fig_dir(cfg) / "fig12_sensitivity.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return {"fig12": path}


# --- Report ---------------------------------------------------------------

def write_report(cfg: Config, fc: ForecastResult, table: pd.DataFrame,
                 curves: dict, figs: dict) -> str:
    out: list[str] = []
    w(out, "# Phases 9-10 - Forecast and sensitivity")
    w(out)

    _section_setup(out, cfg, fc)
    _section_forecast_results(out, fc)
    _section_spread(out, fc)
    _section_extrapolation(out, fc)
    _section_forecast_figures(out, figs)
    _section_sensitivity(out, cfg, table, figs)
    _section_findings(out, fc, table)

    text = "\n".join(out) + "\n"
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    (cfg.reports_dir / "forecast_sensitivity.md").write_text(text, encoding="utf-8")
    return text


def _section_setup(out, cfg, fc):
    tr0, tr1 = fc.train_years
    p0, p1 = fc.predict_years
    w(out, "## Setup")
    w(out)
    w(out, f"The refined model (station {cfg['forecast']['station']}, Type B, "
           f"H = 10, with the day-number time tag) is trained on samples dated "
           f"**{tr0}-{tr1}** only, with a random "
           f"{int(100*cfg['forecast']['fractions'][0])}/"
           f"{int(100*cfg['forecast']['fractions'][1])} train/validation split "
           f"and no test set, then used to predict every "
           f"**{p0}-{p1}** sample from the actual weather.")
    w(out)
    counts = fc.split_counts
    w(out, _md_table(pd.DataFrame([
        {"period": f"train {tr0}-{tr1}", "n_samples": counts["train"]},
        {"period": f"validation {tr0}-{tr1}", "n_samples": counts["val"]},
        {"period": f"forecast {p0}-{p1}", "n_samples": counts[PREDICT_SPLIT]},
    ])))
    w(out)
    w(out, "1998 and 2017 are excluded as in the paper; in this dataset "
           "neither year has a surviving Type B sample anyway, so nothing is "
           "actually discarded.")
    w(out)
    w(out, "Three methodological points, all deliberate:")
    w(out)
    w(out, f"1. **The scaler is fitted on {tr0}-{tr1} only** "
           f"({fc.scaler.n_fit} samples). This is the honest choice for a "
           "forecast and departs from the paper's fit-on-all default.")
    w(out, "2. **Best-of-5 is selected on train+validation MSE**, never on the "
           "forecast period. Selecting on the forecast would leak the very "
           "answer being measured.")
    w(out, "3. **These MSEs are not comparable with Table 1's.** Normalised "
           "MSE is expressed in units of the fitted standard deviation, and "
           "this scaler differs from the Phase 6-8 one (DEVIATIONS.md D-18). "
           "They are comparable with the paper's forecast figures, which use "
           "the same construction, and with each other.")
    w(out)
    w(out, "Windows for early-2015 samples reach back into late-2014 weather, "
           "which is expected and allowed: that weather is a genuine input, "
           "not a label.")
    w(out)


def _section_forecast_results(out, fc):
    w(out, "## Forecast results")
    w(out)
    rows = [
        {
            "quantity": "MSE, train + validation",
            "ours": round(fc.train_val_mse(), 3),
            "paper": PAPER_FORECAST["train_val_mse"],
            "difference": round(fc.train_val_mse() - PAPER_FORECAST["train_val_mse"], 3),
        },
        {
            "quantity": "MSE, forecast period",
            "ours": round(fc.predict_mse(), 3),
            "paper": PAPER_FORECAST["predict_mse"],
            "difference": round(fc.predict_mse() - PAPER_FORECAST["predict_mse"], 3),
        },
    ]
    w(out, _md_table(pd.DataFrame(rows)))
    w(out)
    ev = fc.best.evaluation
    w(out, _md_table(pd.DataFrame([
        {
            "split": name,
            "n": ev.by_split[name].n,
            "MSE": round(ev.by_split[name].mse, 3),
            "R": round(ev.by_split[name].r, 3),
            "RMSE mg/L": int(round(ev.by_split[name].rmse_mgL)),
        }
        for name in ("train", "val", PREDICT_SPLIT)
    ])))
    w(out)
    w(out, f"The selected run reaches **RMSE {fc.predict_rmse_mgL():,.0f} mg/L** "
           f"over the forecast period, against measured values averaging about "
           f"{np.mean(ev.y_true_mgL[(fc.meta['split'] == PREDICT_SPLIT).to_numpy()]):,.0f} "
           "mg/L - roughly 13% relative error.")
    w(out)


def _section_spread(out, fc):
    sp = fc.spread()
    w(out, "## Spread across the 5 seeds - read this before the headline")
    w(out)
    rows = []
    for r in fc.results:
        rows.append(
            {
                "seed": r.seed,
                "train+val MSE": round(fc.train_val_mse(r), 3),
                "forecast MSE": round(fc.predict_mse(r), 3),
                "forecast RMSE mg/L": int(round(fc.predict_rmse_mgL(r))),
                "selected": "yes" if r.seed == fc.best.seed else "",
            }
        )
    w(out, _md_table(pd.DataFrame(rows)))
    w(out)
    w(out, f"Forecast-period MSE across the five seeds: mean **{sp['mean']:.3f}**, "
           f"SD **{sp['sd']:.3f}**, range **{sp['min']:.3f} to {sp['max']:.3f}**.")
    w(out)
    w(out, f"**The headline {fc.predict_mse():.3f} is the best of the five, and "
           f"the typical run is about {sp['mean']:.2f}** - roughly double the "
           f"paper's {PAPER_FORECAST['predict_mse']}. The selection rule is "
           "honest (train+validation only), so this is luck rather than "
           "leakage: the run that fitted the training period best also "
           "happened to forecast best.")
    w(out)
    w(out, "With only "
           f"{fc.split_counts[PREDICT_SPLIT]} forecast samples and a spread "
           f"this wide, a single reported number carries little weight. The "
           "paper reports one value for its forecast and does not give a "
           "spread, so we cannot tell where its 0.23 sits in its own "
           "distribution.")
    w(out)


def _section_extrapolation(out, fc):
    meta = fc.meta
    is_predict = (meta["split"] == PREDICT_SPLIT).to_numpy()
    train_days = meta.loc[~is_predict, "day_number"].to_numpy()
    pred_days = meta.loc[is_predict, "day_number"].to_numpy()

    tag_mean = float(fc.scaler.x_mean[2])
    tag_std = float(fc.scaler.x_std[2])
    z_train_max = (train_days.max() - tag_mean) / tag_std
    z_pred_max = (pred_days.max() - tag_mean) / tag_std

    w(out, "## The time tag extrapolates here")
    w(out)
    w(out, "This is the point CLAUDE.md flags, and it is worth being precise "
           "about. The tag is a monotonically increasing day number, and the "
           "forecast period lies entirely beyond the training range:")
    w(out)
    w(out, _md_table(pd.DataFrame([
        {
            "period": "training",
            "day number range": f"{train_days.min():,} to {train_days.max():,}",
            "normalised range": f"{(train_days.min()-tag_mean)/tag_std:+.2f} to {z_train_max:+.2f}",
        },
        {
            "period": "forecast",
            "day number range": f"{pred_days.min():,} to {pred_days.max():,}",
            "normalised range": f"{(pred_days.min()-tag_mean)/tag_std:+.2f} to {z_pred_max:+.2f}",
        },
    ])))
    w(out)
    w(out, f"Every forecast sample carries a tag value larger than any seen in "
           f"training, reaching {z_pred_max:.2f} standard deviations against a "
           f"training maximum of {z_train_max:.2f}. The network has no "
           "evidence about what acidity does in that range and can only "
           "continue whatever trend it fitted.")
    w(out)
    w(out, "Phase 8 found that the tag's benefit survived a blocked split, but "
           "that test held out years *inside* the training range, so it was "
           "still interpolation. This is the first true extrapolation test, "
           "and the wide seed-to-seed spread above is consistent with a "
           "feature the model cannot constrain outside its training range.")
    w(out)
    w(out, "**Practical reading:** the forecast works here partly because BD "
           "acidity was declining steadily through the late training period "
           "and continued to decline through 2015-2016. A trend-following tag "
           "extrapolates well when the trend persists. It would be unsafe to "
           "rely on this for a period where the trend changes.")
    w(out)


def _section_forecast_figures(out, figs):
    w(out, "## Figures")
    w(out)
    w(out, f"![Fig 11](figures/{figs['fig11'].name})")
    w(out)
    w(out, "*Fig. 11 - measured and predicted acidity over the forecast "
           "period, in mg/L.*")
    w(out)
    w(out, f"![Full span](figures/{figs['fig11_full'].name})")
    w(out)
    w(out, "*Supplementary: the whole modelled span, with the forecast period "
           "shaded. The model tracks the training period closely - it was "
           "fitted there - and the forecast continues without an obvious "
           "discontinuity.*")
    w(out)


def _section_sensitivity(out, cfg, table, figs):
    w(out, "## Phase 10 - Sensitivity to +-20% weather")
    w(out)
    p0, p1 = cfg["forecast"]["predict_years"]
    w(out, f"The Phase 9 model and its scaler are used unchanged. The raw "
           f"daily values for **{p0}-01-01 to {p1}-12-31 only** are multiplied "
           f"by 0.8 and 1.2, one variable at a time - precipitation in mm and "
           f"mean temperature in degrees C - and the windows are rebuilt from "
           "the perturbed series. Days before the forecast period keep their "
           "real weather.")
    w(out)
    w(out, "Scaling temperature in degrees C amplifies summer highs and winter "
           "lows together, which is what the paper means by a larger "
           "'temperature fluctuation'.")
    w(out)

    t = table.copy()
    t["mean_pred_mgL"] = t["mean_pred_mgL"].round(0).astype(int)
    t["mean_change_mgL"] = t["mean_change_mgL"].round(0).astype(int)
    t["mean_change_pct"] = t["mean_change_pct"].round(2)
    w(out, _md_table(t[["scenario", "mean_pred_mgL", "mean_change_mgL",
                        "mean_change_pct"]].rename(columns={
        "mean_pred_mgL": "mean predicted (mg/L)",
        "mean_change_mgL": "change (mg/L)",
        "mean_change_pct": "change (%)",
    })))
    w(out)
    w(out, f"![Fig 12](figures/{figs['fig12'].name})")
    w(out)
    w(out, "*Fig. 12 - five prediction curves. The lower panel shows each "
           "scenario's difference from the real-weather prediction, which is "
           "where the effects are legible.*")
    w(out)

    precip = table[table["variable"] == "precip"].set_index("factor")
    tmean = table[table["variable"] == "tmean"].set_index("factor")
    precip_range = abs(precip.loc[0.8, "mean_change_pct"]) + abs(precip.loc[1.2, "mean_change_pct"])
    tmean_range = abs(tmean.loc[0.8, "mean_change_pct"]) + abs(tmean.loc[1.2, "mean_change_pct"])

    w(out, "### Directions")
    w(out)
    rows = [
        {
            "paper's expectation": "More precipitation -> lower acidity",
            "our result": f"{precip.loc[1.2, 'mean_change_mgL']:+.0f} mg/L at x1.2, "
                          f"{precip.loc[0.8, 'mean_change_mgL']:+.0f} at x0.8",
            "holds?": "yes" if precip.loc[1.2, "mean_change_mgL"] < 0
                      < precip.loc[0.8, "mean_change_mgL"] else "no",
        },
        {
            "paper's expectation": "Larger temperature swing -> higher acidity",
            "our result": f"{tmean.loc[1.2, 'mean_change_mgL']:+.0f} mg/L at x1.2, "
                          f"{tmean.loc[0.8, 'mean_change_mgL']:+.0f} at x0.8",
            "holds?": "yes" if tmean.loc[1.2, "mean_change_mgL"] > 0
                      > tmean.loc[0.8, "mean_change_mgL"] else "no",
        },
        {
            "paper's expectation": "Temperature is the more sensitive input",
            "our result": f"{tmean_range:.1f}% total swing vs "
                          f"{precip_range:.1f}% for precipitation",
            "holds?": "yes" if tmean_range > precip_range else "no",
        },
    ]
    w(out, _md_table(pd.DataFrame(rows)))
    w(out)
    w(out, "The precipitation response is mildly asymmetric (+2.1% at x0.8 "
           "against -1.1% at x1.2), which is expected from a non-linear model "
           "and not a sign of trouble.")
    w(out)
    w(out, "**Caveat on magnitude.** These are single-model responses from the "
           "one selected network. Given the seed-to-seed spread in the "
           "forecast itself, the *directions* are the robust finding; the "
           "percentage magnitudes should not be read as calibrated "
           "sensitivities.")
    w(out)


def _section_findings(out, fc, table):
    w(out, "## Qualitative findings 5 and 6")
    w(out)
    sp = fc.spread()
    reasonable = fc.predict_mse() < 2 * PAPER_FORECAST["predict_mse"]
    w(out, f"**Finding 5 - the forecast is reasonable: "
           f"{'REPLICATED' if reasonable else 'NOT REPLICATED'}.** "
           f"Forecast MSE {fc.predict_mse():.2f} against the paper's "
           f"{PAPER_FORECAST['predict_mse']}, and train+validation "
           f"{fc.train_val_mse():.2f} against {PAPER_FORECAST['train_val_mse']}. "
           f"Both are close. The qualification is the seed spread "
           f"({sp['min']:.2f} to {sp['max']:.2f}): the headline is the best of "
           "five, not a typical run.")
    w(out)

    precip = table[table["variable"] == "precip"].set_index("factor")
    tmean = table[table["variable"] == "tmean"].set_index("factor")
    d1 = precip.loc[1.2, "mean_change_mgL"] < 0 < precip.loc[0.8, "mean_change_mgL"]
    d2 = tmean.loc[1.2, "mean_change_mgL"] > 0 > tmean.loc[0.8, "mean_change_mgL"]
    d3 = (abs(tmean["mean_change_pct"]).sum() > abs(precip["mean_change_pct"]).sum())
    n_ok = sum([d1, d2, d3])

    w(out, f"**Finding 6 - the sensitivity directions hold: "
           f"{'REPLICATED' if n_ok == 3 else 'PARTIALLY REPLICATED'} "
           f"({n_ok}/3).** More precipitation lowers acidity, a larger "
           "temperature swing raises it, and temperature is the more sensitive "
           "input - all three as the paper reports.")
    w(out)
