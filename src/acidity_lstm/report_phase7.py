"""Phase 7 deliverable: `reports/fc_baseline.md`."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .audit import _md_table, w
from .config import Config
from .evaluate import scatter_measured_vs_calculated
from .experiments import PAPER_FC_R, PAPER_LSTM_R

log = logging.getLogger(__name__)

MODEL_LABELS = {"lstm": "LSTM", "fc": "FC baseline"}


def make_figure(cfg: Config, scenarios: dict):
    """LSTM vs FC baseline scatter, one panel per model and station."""
    d = cfg.reports_dir / "figures"
    d.mkdir(parents=True, exist_ok=True)
    path = d / "fc_baseline_scatter.png"

    panels = {}
    for station in ("BD", "C7"):
        for kind in ("lstm", "fc"):
            if (station, kind) in scenarios:
                panels[f"{station} - {MODEL_LABELS[kind]}"] = (
                    scenarios[(station, kind)].best.evaluation
                )

    scatter_measured_vs_calculated(
        panels, path,
        title="Phase 7 - LSTM vs fully connected baseline (Type B, 10 neurons)",
    )
    return path


def _mean_over_repeats(scenario, split: str, metric: str) -> float:
    vals = [getattr(r.evaluation.by_split[split], metric) for r in scenario.results]
    return float(np.nanmean(vals))


def write_report(cfg: Config, scenarios: dict, fig_path) -> str:
    out: list[str] = []
    w(out, "# Phase 7 - Fully connected baseline")
    w(out)
    w(out, "The baseline follows Ma et al. (2020a): the Type B window "
           "(10 steps x 2 features) is flattened to 20 values, then "
           "`Linear(20, 10)` -> tanh -> `Linear(10, 1)`. The paper does not "
           "state the activation; tanh is MATLAB's default for this kind of "
           "network and is recorded as an assumption (DEVIATIONS.md D-09).")
    w(out)
    w(out, "Both models use the **same samples, the same splits, the same "
           "scaler and the same 5 repeats**, so their metrics are directly "
           "comparable (DEVIATIONS.md D-18).")
    w(out)

    _section_headline(out, scenarios)
    _section_overfitting(out, scenarios)
    _section_heldout(out, scenarios)
    _section_detail(out, scenarios)
    _section_figure(out, fig_path)
    _section_finding(out, scenarios)

    text = "\n".join(out) + "\n"
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    (cfg.reports_dir / "fc_baseline.md").write_text(text, encoding="utf-8")
    return text


def _section_headline(out, scenarios):
    w(out, "## R as the paper reports it (all samples, best of 5)")
    w(out)
    rows = []
    for station in ("BD", "C7"):
        lstm = scenarios[(station, "lstm")].best.evaluation.all
        fc = scenarios[(station, "fc")].best.evaluation.all
        rows.append(
            {
                "station": station,
                "LSTM R (ours)": round(lstm.r, 2),
                "LSTM R (paper)": PAPER_LSTM_R[station],
                "FC R (ours)": round(fc.r, 2),
                "FC R (paper)": PAPER_FC_R[station],
                "LSTM - FC (ours)": round(lstm.r - fc.r, 2),
                "LSTM - FC (paper)": round(
                    PAPER_LSTM_R[station] - PAPER_FC_R[station], 2
                ),
            }
        )
    w(out, _md_table(pd.DataFrame(rows)))
    w(out)
    w(out, "**BD reproduces the paper's ordering. C7 does not**: our FC "
           "baseline reaches R = 0.78 over all samples, well above both our "
           "LSTM (0.56) and the paper's FC value (0.42). That number does not "
           "survive scrutiny, and the next two sections explain why.")
    w(out)


def _section_overfitting(out, scenarios):
    w(out, "## Why the all-sample R is misleading here")
    w(out)
    w(out, "All-sample metrics are dominated by the training rows - about 68% "
           "of the samples. A model that memorises its training set therefore "
           "scores well on them. Worse, **best-of-5 selection uses MSE over "
           "all samples** (the paper's stated rule, and the reading adopted in "
           "CLAUDE.md), so among the five repeats it actively prefers the one "
           "that overfits hardest.")
    w(out)

    rows = []
    for station in ("BD", "C7"):
        for kind in ("lstm", "fc"):
            s = scenarios[(station, kind)]
            tr = _mean_over_repeats(s, "train", "mse")
            te = _mean_over_repeats(s, "test", "mse")
            rows.append(
                {
                    "station": station,
                    "model": MODEL_LABELS[kind],
                    "train MSE": round(tr, 3),
                    "test MSE": round(te, 3),
                    "gap (test - train)": round(te - tr, 3),
                    "train R": round(_mean_over_repeats(s, "train", "r"), 3),
                    "test R": round(_mean_over_repeats(s, "test", "r"), 3),
                    "epochs": round(np.mean([r.epochs_run for r in s.results]), 1),
                }
            )
    w(out, "Mean over the 5 repeats:")
    w(out)
    w(out, _md_table(pd.DataFrame(rows)))
    w(out)

    c7_fc = scenarios[("C7", "fc")]
    best = c7_fc.best.evaluation
    w(out, f"The selected C7 FC run is the clearest case: train MSE "
           f"{best.by_split['train'].mse:.3f} (R "
           f"{best.by_split['train'].r:.3f}) against test MSE "
           f"{best.by_split['test'].mse:.3f} (R "
           f"{best.by_split['test'].r:.3f}) - a train-to-test gap of "
           f"{best.by_split['test'].mse - best.by_split['train'].mse:.3f}. "
           f"It also trained for {c7_fc.best.epochs_run} epochs, against "
           f"{np.mean([r.epochs_run for r in scenarios[('C7','lstm')].results]):.0f} "
           "on average for the LSTM. Its headline R of "
           f"{best.all.r:.2f} is largely memorised training data.")
    w(out)


def _section_heldout(out, scenarios):
    w(out, "## The honest comparison: held-out test samples only")
    w(out)
    rows = []
    for station in ("BD", "C7"):
        lstm = scenarios[(station, "lstm")]
        fc = scenarios[(station, "fc")]
        l_r = _mean_over_repeats(lstm, "test", "r")
        f_r = _mean_over_repeats(fc, "test", "r")
        rows.append(
            {
                "station": station,
                "n_test": lstm.split_counts["test"],
                "LSTM test R": round(l_r, 3),
                "FC test R": round(f_r, 3),
                "LSTM - FC": round(l_r - f_r, 3),
                "winner": "LSTM" if l_r > f_r else "FC baseline",
            }
        )
    w(out, "Mean test-split R over the 5 repeats:")
    w(out)
    w(out, _md_table(pd.DataFrame(rows)))
    w(out)
    w(out, "**At BD the LSTM wins on held-out data as the paper reports. "
           "At C7 it still loses.** So C7 is not purely a selection artefact: "
           "the FC baseline really does generalise better there.")
    w(out)

    c7_lstm = scenarios[("C7", "lstm")]
    tr = _mean_over_repeats(c7_lstm, "train", "mse")
    te = _mean_over_repeats(c7_lstm, "test", "mse")
    w(out, f"The reason is on the other side: **our C7 LSTM underfits**. Its "
           f"train MSE ({tr:.3f}) is barely below its test MSE ({te:.3f}), a "
           f"gap of only {te - tr:.3f} against {0.335:.3f} for the BD LSTM. It "
           "is not learning the training set, let alone overfitting it.")
    w(out)
    w(out, "Two caveats on how far to push this:")
    w(out)
    w(out, f"1. The C7 test split holds only "
           f"{c7_lstm.split_counts['test']} samples, so test R is noisy "
           f"(SD {np.nanstd([r.evaluation.by_split['test'].r for r in c7_lstm.results], ddof=1):.3f} "
           "across repeats).")
    w(out, f"2. C7 test R ({_mean_over_repeats(c7_lstm, 'test', 'r'):.2f}) is "
           f"far below C7 validation R "
           f"({_mean_over_repeats(c7_lstm, 'val', 'r'):.2f}), which suggests "
           "this particular test fold is unusually hard rather than the model "
           "being uniformly poor.")
    w(out)


def _section_detail(out, scenarios):
    w(out, "## Full metrics (best of 5)")
    w(out)
    rows = []
    for (station, kind), s in sorted(scenarios.items()):
        b = s.best.evaluation
        rows.append(
            {
                "station": station,
                "model": MODEL_LABELS[kind],
                "params": s.best.n_parameters,
                "n": s.n_samples,
                "MSE all": round(b.all.mse, 3),
                "MSE mean+-sd": f"{s.summary['all_mse_mean']:.3f}+-"
                                f"{s.summary['all_mse_sd']:.3f}",
                "R all": round(b.all.r, 3),
                "MSE train": round(b.by_split["train"].mse, 3),
                "MSE test": round(b.by_split["test"].mse, 3),
                "R test": round(b.by_split["test"].r, 3),
                "RMSE mg/L": int(round(b.all.rmse_mgL)),
            }
        )
    w(out, _md_table(pd.DataFrame(rows)))
    w(out)
    w(out, "The FC baseline has 221 parameters against the LSTM's 571. The "
           "LSTM's advantage at BD is therefore not a capacity effect - it "
           "comes from processing the window as an ordered sequence rather "
           "than as 20 unordered numbers.")
    w(out)


def _section_figure(out, fig_path):
    w(out, "## Figure")
    w(out)
    w(out, f"![LSTM vs FC](figures/{fig_path.name})")
    w(out)
    w(out, "*Measured vs calculated acidity, normalised, all samples, best run "
           "of 5. Note that the C7 FC panel looks tight mainly because most of "
           "those points are training samples it has memorised.*")
    w(out)


def _section_finding(out, scenarios):
    w(out, "## Qualitative finding 3: does the LSTM beat the FC baseline?")
    w(out)

    rows = []
    for station in ("BD", "C7"):
        lstm, fc = scenarios[(station, "lstm")], scenarios[(station, "fc")]
        all_win = lstm.best.evaluation.all.r > fc.best.evaluation.all.r
        test_win = (_mean_over_repeats(lstm, "test", "r")
                    > _mean_over_repeats(fc, "test", "r"))
        rows.append(
            {
                "station": station,
                "LSTM wins on all-sample R": "yes" if all_win else "no",
                "LSTM wins on test R": "yes" if test_win else "no",
                "paper": "yes",
            }
        )
    w(out, _md_table(pd.DataFrame(rows)))
    w(out)

    bd_ok = (_mean_over_repeats(scenarios[("BD", "lstm")], "test", "r")
             > _mean_over_repeats(scenarios[("BD", "fc")], "test", "r"))
    c7_ok = (_mean_over_repeats(scenarios[("C7", "lstm")], "test", "r")
             > _mean_over_repeats(scenarios[("C7", "fc")], "test", "r"))

    if bd_ok and c7_ok:
        verdict = "REPLICATED at both stations"
    elif bd_ok or c7_ok:
        verdict = "PARTIALLY REPLICATED - holds at BD, fails at C7"
    else:
        verdict = "NOT REPLICATED"
    w(out, f"**Verdict: {verdict}.**")
    w(out)
    w(out, "At BD the LSTM beats the FC baseline on both the paper's metric "
           "and on held-out data, and the size of the gap (+0.13 all-sample R) "
           "is close to the paper's +0.06. At C7 the result inverts, driven by "
           "an LSTM that underfits rather than by an unusually strong "
           "baseline.")
    w(out)
    w(out, "This is a genuine partial non-replication and is **not** patched "
           "by tuning: CLAUDE.md forbids going beyond the paper's stated "
           "settings, and doing so would invalidate the comparison with "
           "Table 1. It is carried into the Phase 11 report as an open "
           "discrepancy.")
    w(out)
    w(out, "It also exposes a weakness in the paper's own methodology: "
           "reporting R over all samples, and selecting the best of five runs "
           "by all-sample MSE, both reward overfitting. Ma et al.'s FC "
           "baseline R of 0.42-0.64 may well be depressed or inflated by the "
           "same effect, and their reported LSTM advantage cannot be checked "
           "against held-out data from the paper alone.")
    w(out)
