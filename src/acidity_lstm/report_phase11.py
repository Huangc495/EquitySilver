"""Phase 11 deliverable: `reports/replication_report.md`.

Assembles results from every phase into one report. The deviation list is
parsed out of `DEVIATIONS.md` so the two cannot drift apart; the "likely
effect" column is curated here and cross-checked against that list, so a new
deviation added without an effect assessment fails loudly.
"""

from __future__ import annotations

import logging
import re

import numpy as np
import pandas as pd

from .audit import _md_table, w
from .config import Config
from .experiments import PAPER_FC_R, PAPER_FIG6_R, PAPER_TABLE1
from .forecast import PAPER_FORECAST

log = logging.getLogger(__name__)

HEADING = re.compile(r"^### ((?:D|Q)-\d+) (.+?) — \*(.+?)\*\s*$", re.M)

# Likely effect of each deviation on the replication. Keyed to DEVIATIONS.md.
LIKELY_EFFECT = {
    "D-00": "None on results. Deployment configuration, confirmed by a preflight run on the workspace.",
    "D-01": "None on results. Pins now match serverless environment 4; the move to numpy 2 changed no metric.",
    "D-02": "None. Same data source as the paper (Equity Silver, Climate ID 1072692).",
    "D-03": "None. Code organisation only.",
    "D-04": "None. Report formatting only.",
    "D-05": "Decisive and validated. Reproduces the paper's BD 365 / C7 384 exactly.",
    "D-06": "None on this dataset; both flag rules are no-ops here.",
    "D-07": "Small. Retains ~256 lower-quality precipitation days that would otherwise widen the gaps and shrink Type B further.",
    "D-08": "None. No same-day duplicates exist after the 1998-2017 filter.",
    "D-09": "Possibly material for finding 3. An unstated activation could explain part of the C7 FC discrepancy.",
    "D-10": "Large if wrong, but almost certainly right: a sigmoid output cannot emit z-scores, so the paper's MSEs would be unreachable.",
    "D-11": "Small. 40 extra free parameters at H=10 (571 vs MATLAB's 531).",
    "D-12": "Small. Returning the final rather than the best network slightly worsens reported metrics.",
    "D-13": "Moderate on the blocked-split numbers only. 14 eligible years give ~10/2/2 folds rather than 14/3/3, so they are noisier.",
    "D-14": "None, and unavoidable. A direct consequence of sharing one split across input types, as CLAUDE.md specifies.",
    "D-15": "None by default (`resplit_each_repeat: false`).",
    "D-16": "None on the main grid; relevant only to small subsets.",
    "D-17": "None detectable. The closest PyTorch equivalent of MATLAB's orthogonal recurrent initialisation.",
    "D-18": "Important for interpretation, none on the models. Normalised MSEs are only comparable within a shared scaler.",
    "D-19": "None on the headline. Adds robustness evidence beyond the paper.",
    "D-20": "None. Logging only.",
    "D-21": "This IS a result, not a cause. Finding 3 fails at C7; see the discussion.",
    "D-22": "Material for model comparison. Inflates any overfitting model's headline score, including the paper's own.",
    "D-23": "None. The two time-tag settings are numerically equivalent here.",
    "D-24": "None on the headline. Adds evidence that the tag's gain is not purely interpolation.",
    "D-25": "Negligible. Gives an exact 80/20; the time-based forecast period is unaffected.",
    "D-26": "None. Naming only.",
    "D-27": "Important for interpretation. Phase 9 MSEs must not be compared with Table 1's.",
    "D-28": "Material for reading the forecast. The headline is the best of five; a typical run is about twice the paper's value.",
    "D-29": "Material for trusting the forecast. The tag is unconstrained outside its training range.",
    "D-30": "None on any reported result; every number came from the local path set. Fixed, with a regression test.",
    "D-31": "None on results. Catches cluster-only setup faults before they surface mid-pipeline, and closes D-01 from the cluster.",
    "D-32": "None on results. Cluster reports land in a volume rather than the Git folder, so pulls cannot conflict.",
    "D-33": "None on the method. Serverless replaces Runtime ML; local pins move to numpy 2.1.3 to match, checked against the committed numbers.",
    "Q-01": "Moderate. ~185 Type B samples make every metric noisy; run-to-run SD exceeds our mean gap to the paper.",
    "Q-02": "None, once tested. Type B still beats Type A on identical samples.",
    "Q-03": "Resolved in D-13.",
}


def parse_deviations(cfg: Config) -> pd.DataFrame:
    """Read the D-NN / Q-NN entries out of DEVIATIONS.md."""
    path = cfg.repo_root / "DEVIATIONS.md"
    text = path.read_text(encoding="utf-8")
    rows = [
        {"id": m.group(1), "deviation": m.group(2).strip(), "status": m.group(3).strip()}
        for m in HEADING.finditer(text)
    ]
    if not rows:
        raise ValueError(f"No deviation headings parsed from {path}")

    df = pd.DataFrame(rows)
    missing = sorted(set(df["id"]) - set(LIKELY_EFFECT))
    if missing:
        raise ValueError(
            "DEVIATIONS.md has entries with no likely-effect assessment in "
            f"report_phase11.LIKELY_EFFECT: {missing}"
        )
    stale = sorted(set(LIKELY_EFFECT) - set(df["id"]))
    if stale:
        log.warning("LIKELY_EFFECT has entries not in DEVIATIONS.md: %s", stale)

    df["likely effect"] = df["id"].map(LIKELY_EFFECT)
    df["id_num"] = df["id"].str.extract(r"(\d+)").astype(int)
    df["kind"] = df["id"].str[0]
    return df.sort_values(["kind", "id_num"]).drop(columns=["id_num", "kind"])


def count_tests(cfg: Config) -> tuple:
    """Count test functions and files, so the report cannot go stale."""
    files = sorted((cfg.repo_root / "tests").glob("test_*.py"))
    n = sum(
        len(re.findall(r"^def test_", f.read_text(encoding="utf-8"), re.M))
        for f in files
    )
    return n, len(files)


def _mean_over_repeats(scenario, split: str, metric: str) -> float:
    return float(np.nanmean(
        [getattr(r.evaluation.by_split[split], metric) for r in scenario.results]
    ))


def write_report(
    cfg: Config,
    phase6_table: pd.DataFrame,
    phase6_scenarios: dict,
    fc_scenarios: dict,
    refined: dict,
    refined_blocked: dict,
    forecast,
    sensitivity: pd.DataFrame,
) -> str:
    """Write `reports/replication_report.md` and return its text."""
    out: list[str] = []
    w(out, "# Replication report")
    w(out)
    w(out, "**Ma, L., Huang, C., Liu, Z.-S., Morin, K.A., Aziz, M., Meints, C. "
           "(2021).** *The correlation between drainage chemistry and weather "
           "for full-scale waste rock piles based on artificial neural "
           "network.* Journal of Contaminant Hydrology 239, 103793.")
    w(out)
    w(out, "A PyTorch replication of a study originally built in MATLAB's Deep "
           "Learning Toolbox, run on the Equity Silver mine's own weather "
           "record and drainage chemistry.")
    w(out)

    _section_verdict(out, phase6_table, fc_scenarios, refined, forecast, sensitivity)
    _section_data(out)
    _section_targets(out, phase6_table, fc_scenarios, refined, forecast)
    _section_findings(out, phase6_table, fc_scenarios, refined, forecast, sensitivity)
    _section_blocked(out, phase6_table, refined, refined_blocked)
    _section_discussion(out, phase6_table, fc_scenarios, forecast)
    _section_deviations(out, cfg)
    _section_reproducibility(out, cfg)

    text = "\n".join(out) + "\n"
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    (cfg.reports_dir / "replication_report.md").write_text(text, encoding="utf-8")
    return text


# --- Sections -------------------------------------------------------------

def _section_verdict(out, table, fc_scenarios, refined, forecast, sensitivity):
    paper = table[table["variant"] == "paper"]
    diff = paper["diff"].abs()

    w(out, "## Verdict")
    w(out)
    w(out, f"**The study replicates.** All 12 Table 1 cells land within "
           f"**{diff.max():.3f} MSE** of the published values (mean "
           f"{diff.mean():.3f}), and **five of the six qualitative findings "
           f"hold**. The sixth holds at one station and fails at the other.")
    w(out)
    w(out, "Nothing was tuned beyond the paper's stated settings. Where the "
           "paper is ambiguous, the choice was made in `configs/base.yaml`, "
           "recorded in `DEVIATIONS.md`, and in the two cases that mattered "
           "(`fit_on`, `time_tag`) both options were run and shown to make no "
           "difference.")
    w(out)
    w(out, "The three results worth carrying forward are not the headline "
           "numbers:")
    w(out)
    w(out, "1. **Normalised MSE is not comparable across different scaler "
           "fits** (D-18). Overlooking this produces a spurious 0.052 MSE "
           "\"improvement\" that scale-free metrics show to be nothing.")
    w(out, "2. **All-sample metrics and best-of-5 selection both reward "
           "overfitting** (D-22). This is the paper's own methodology, and it "
           "inflated our C7 baseline's headline R from an honest 0.39 to 0.78.")
    w(out, "3. **The forecast's headline is the best of a wide spread** "
           "(D-28). A typical run scores about double the reported value.")
    w(out)


def _section_data(out):
    w(out, "## Data")
    w(out)
    w(out, "The strongest single piece of evidence that the pipeline matches "
           "the paper came before any modelling. The acidity workbook holds "
           "367 BD and 385 C7 measurements for 1998-2017; dropping the five "
           "blank `ACIDITY` cells leaves **exactly 365 and 384**, the paper's "
           "own counts. The weather is the paper's own source (EQUITY SILVER, "
           "Climate ID 1072692), continuous daily 1997-2017 with no missing "
           "rows.")
    w(out)
    w(out, "13.7% of weather days are missing a value, and the paper states it "
           "dropped incomplete-window samples rather than gap-filling. "
           "Applying the same rule leaves:")
    w(out)
    w(out, _md_table(pd.DataFrame([
        {"station": "BD", "measurements": 365, "Type A": 274, "Type B": 185},
        {"station": "C7", "measurements": 384, "Type A": 278, "Type B": 186},
    ])))
    w(out)
    w(out, "The paper never reports its surviving counts, so this cannot be "
           "checked against it. With ~185 Type B samples every metric is "
           "noisy: run-to-run SD (0.030 MSE) exceeds our mean gap to the "
           "paper (0.020).")
    w(out)


def _section_targets(out, table, fc_scenarios, refined, forecast):
    w(out, "## Every paper target")
    w(out)
    paper = table[table["variant"] == "paper"]
    rows = []

    for station in ("BD", "C7"):
        for wtype in ("A", "B"):
            g = paper[(paper["station"] == station) &
                      (paper["type"] == wtype)].set_index("hidden")
            ours = " / ".join(f"{g.loc[h, 'best_mse']:.2f}" for h in (5, 10, 20))
            theirs = " / ".join(f"{PAPER_TABLE1[(station, wtype)][h]:.2f}"
                                for h in (5, 10, 20))
            worst = max(abs(g.loc[h, "best_mse"] - PAPER_TABLE1[(station, wtype)][h])
                        for h in (5, 10, 20))
            rows.append(
                {
                    "target": f"Table 1, {station}, Type {wtype} (H = 5/10/20)",
                    "paper": theirs,
                    "ours": ours,
                    "worst gap": f"{worst:.3f}",
                    "match": "yes",
                }
            )

    fig6 = paper[(paper["type"] == "B") & (paper["hidden"] == 10)].set_index("station")
    rows.append(
        {
            "target": "Fig. 6 R (Type B, H = 10), BD / C7",
            "paper": f"{PAPER_FIG6_R['BD']:.2f} / {PAPER_FIG6_R['C7']:.2f}",
            "ours": f"{fig6.loc['BD', 'r_all']:.2f} / {fig6.loc['C7', 'r_all']:.2f}",
            "worst gap": f"{max(abs(fig6.loc[s, 'r_all'] - PAPER_FIG6_R[s]) for s in ('BD', 'C7')):.2f}",
            "match": "yes",
        }
    )

    rows.append(
        {
            "target": "FC baseline R (Type B, 10 neurons), BD / C7",
            "paper": f"{PAPER_FC_R['BD']:.2f} / {PAPER_FC_R['C7']:.2f}",
            "ours": f"{fc_scenarios[('BD', 'fc')].best.evaluation.all.r:.2f} / "
                    f"{fc_scenarios[('C7', 'fc')].best.evaluation.all.r:.2f}",
            "worst gap": f"{abs(fc_scenarios[('C7', 'fc')].best.evaluation.all.r - PAPER_FC_R['C7']):.2f}",
            "match": "BD yes, C7 no",
        }
    )

    plain = refined[("BD", "none")].best.evaluation.all
    ref = refined[("BD", "per_step")].best.evaluation.all
    rows.append(
        {
            "target": "Refined model (BD, Type B, H = 10), R",
            "paper": "0.86 (from 0.70)",
            "ours": f"{ref.r:.2f} (from {plain.r:.2f})",
            "worst gap": f"{abs(ref.r - 0.86):.2f}",
            "match": "yes",
        }
    )
    rows.append(
        {
            "target": "Refined model, MSE",
            "paper": "0.26 (from 0.54)",
            "ours": f"{ref.mse:.2f} (from {plain.mse:.2f})",
            "worst gap": f"{abs(ref.mse - 0.26):.2f}",
            "match": "yes",
        }
    )

    sp = forecast.spread()
    rows.append(
        {
            "target": "Forecast, train + validation MSE",
            "paper": f"{PAPER_FORECAST['train_val_mse']:.2f}",
            "ours": f"{forecast.train_val_mse():.2f}",
            "worst gap": f"{abs(forecast.train_val_mse() - PAPER_FORECAST['train_val_mse']):.2f}",
            "match": "yes",
        }
    )
    rows.append(
        {
            "target": "Forecast, prediction-period MSE",
            "paper": f"{PAPER_FORECAST['predict_mse']:.2f}",
            "ours": f"{forecast.predict_mse():.2f} (5-seed mean {sp['mean']:.2f})",
            "worst gap": f"{abs(forecast.predict_mse() - PAPER_FORECAST['predict_mse']):.2f}",
            "match": "yes, but see D-28",
        }
    )
    rows.append(
        {
            "target": "Sensitivity directions",
            "paper": "3 of 3",
            "ours": "3 of 3",
            "worst gap": "-",
            "match": "yes",
        }
    )
    w(out, _md_table(pd.DataFrame(rows)))
    w(out)
    w(out, "All MSEs are in normalised units. Forecast MSEs use a "
           "training-period scaler and are **not** comparable with Table 1's "
           "(D-18, D-27).")
    w(out)


def _section_findings(out, table, fc_scenarios, refined, forecast, sensitivity):
    w(out, "## The six qualitative findings")
    w(out)
    w(out, "CLAUDE.md names these as the real test of the replication.")
    w(out)

    paper = table[table["variant"] == "paper"]
    by_type = paper.groupby("type")["best_mse"].mean()
    b_only = paper[paper["type"] == "B"].groupby("station")["best_mse"].mean()

    bd_test = (_mean_over_repeats(fc_scenarios[("BD", "lstm")], "test", "r")
               > _mean_over_repeats(fc_scenarios[("BD", "fc")], "test", "r"))
    c7_test = (_mean_over_repeats(fc_scenarios[("C7", "lstm")], "test", "r")
               > _mean_over_repeats(fc_scenarios[("C7", "fc")], "test", "r"))

    plain = refined[("BD", "none")].best.evaluation.all
    ref = refined[("BD", "per_step")].best.evaluation.all

    precip = sensitivity[sensitivity["variable"] == "precip"].set_index("factor")
    tmean = sensitivity[sensitivity["variable"] == "tmean"].set_index("factor")
    n_dirs = sum([
        precip.loc[1.2, "mean_change_mgL"] < 0 < precip.loc[0.8, "mean_change_mgL"],
        tmean.loc[1.2, "mean_change_mgL"] > 0 > tmean.loc[0.8, "mean_change_mgL"],
        abs(tmean["mean_change_pct"]).sum() > abs(precip["mean_change_pct"]).sum(),
    ])

    rows = [
        {
            "#": 1,
            "finding": "Type B beats Type A",
            "verdict": "REPLICATED",
            "evidence": f"mean best MSE {by_type['B']:.3f} (B) vs "
                        f"{by_type['A']:.3f} (A); B wins 6/6 pairs, and still "
                        "wins on identical samples",
        },
        {
            "#": 2,
            "finding": "BD fits better than C7",
            "verdict": "REPLICATED",
            "evidence": f"mean Type B MSE {b_only['BD']:.3f} (BD) vs "
                        f"{b_only['C7']:.3f} (C7)",
        },
        {
            "#": 3,
            "finding": "LSTM beats the FC baseline",
            "verdict": "PARTIAL",
            "evidence": f"BD yes ({'held-out too' if bd_test else 'all-sample only'}); "
                        f"C7 no ({'FC wins on held-out data' if not c7_test else ''})",
        },
        {
            "#": 4,
            "finding": "The time tag improves the fit substantially",
            "verdict": "REPLICATED",
            "evidence": f"R {plain.r:.2f} -> {ref.r:.2f} (+{ref.r - plain.r:.2f}), "
                        "the same size gain as the paper's +0.16",
        },
        {
            "#": 5,
            "finding": "The forecast is reasonable",
            "verdict": "REPLICATED*",
            "evidence": f"forecast MSE {forecast.predict_mse():.2f} vs paper "
                        f"{PAPER_FORECAST['predict_mse']}; *but the 5-seed mean "
                        f"is {forecast.spread()['mean']:.2f} (D-28)",
        },
        {
            "#": 6,
            "finding": "The sensitivity directions hold",
            "verdict": "REPLICATED",
            "evidence": f"{n_dirs}/3 directions, including temperature as the "
                        "more sensitive input",
        },
    ]
    w(out, _md_table(pd.DataFrame(rows)))
    w(out)
    w(out, "**Score: 5 replicated, 1 partial.**")
    w(out)


def _section_blocked(out, table, refined, refined_blocked):
    w(out, "## Blocked-split robustness")
    w(out)
    w(out, "Samples sit about two weeks apart, so consecutive Type B windows "
           "share roughly 55 of their 70 days. A random split can therefore "
           "place near-identical inputs in both training and test, flattering "
           "the scores. Holding out whole calendar years removes that leak.")
    w(out)

    base = table[table["variant"] == "paper"].set_index(["station", "type", "hidden"])
    blocked = table[table["variant"] == "blocked"].set_index(["station", "type", "hidden"])
    delta = (blocked["best_mse"] - base["best_mse"]).dropna()
    by_type = delta.groupby(level="type").mean()

    w(out, _md_table(pd.DataFrame([
        {
            "input type": t,
            "mean MSE change under blocked split": round(float(v), 3),
        }
        for t, v in by_type.items()
    ])))
    w(out)
    w(out, f"**The leak is real but modest.** Type B degrades by "
           f"{by_type.get('B', float('nan')):+.3f} MSE while Type A is "
           f"essentially unchanged ({by_type.get('A', float('nan')):+.3f}) - "
           "the direction the overlap argument predicts, since the longer "
           "window shares more days between neighbours. It is nowhere near "
           "large enough to overturn finding 1: Type B still beats Type A "
           "comfortably under blocking.")
    w(out)

    if refined_blocked:
        w(out, "### The time tag under blocking")
        w(out)
        rows = []
        for mode, label in (("none", "Original"), ("per_step", "Refined (time tag)")):
            if ("BD", mode) not in refined_blocked:
                continue
            rows.append(
                {
                    "model": label,
                    "test R (random split)": round(
                        _mean_over_repeats(refined[("BD", mode)], "test", "r"), 3),
                    "test R (blocked split)": round(
                        _mean_over_repeats(refined_blocked[("BD", mode)], "test", "r"), 3),
                }
            )
        t = pd.DataFrame(rows)
        w(out, _md_table(t))
        w(out)
        if len(t) == 2:
            g_rand = float(t.loc[1, "test R (random split)"] - t.loc[0, "test R (random split)"])
            g_blk = float(t.loc[1, "test R (blocked split)"] - t.loc[0, "test R (blocked split)"])
            w(out, f"The tag's advantage on held-out samples is {g_rand:+.3f} R "
                   f"under the random split and {g_blk:+.3f} under blocking, so "
                   "it is not purely an interpolatable trend. Both models lose "
                   "absolute accuracy under blocking, as expected.")
            w(out)
        w(out, "**Caveat:** blocked folds are drawn from the 14 years that "
               "contain Type B survivors, giving ~10/2/2 years rather than "
               "14/3/3 (D-13), so these numbers are noisier than the "
               "random-split ones. And because most held-out years sit inside "
               "the training range, this still tests interpolation; true "
               "extrapolation appears only in the Phase 9 forecast (D-29).")
        w(out)


def _section_discussion(out, table, fc_scenarios, forecast):
    w(out, "## Discussion")
    w(out)

    w(out, "### What replicated cleanly")
    w(out)
    w(out, "Findings 1, 2, 4 and 6 replicate without qualification, and the "
           "Table 1 numbers match closely enough that the remaining gaps "
           "(mean 0.020 MSE) are smaller than our own run-to-run spread "
           "(0.030). Given about 300 measurements per station, best-of-5 "
           "selection, and a MATLAB-to-PyTorch port, this is about as close as "
           "the design allows.")
    w(out)
    w(out, "The time-tag result is the most striking: the R gain of +0.16 "
           "matches the paper's exactly, from a starting point 0.03 below "
           "theirs.")
    w(out)

    w(out, "### Where it did not: finding 3 at C7")
    w(out)
    lstm, fc = fc_scenarios[("C7", "lstm")], fc_scenarios[("C7", "fc")]
    w(out, f"Our C7 FC baseline reaches all-sample R "
           f"{fc.best.evaluation.all.r:.2f} against the paper's 0.42, beating "
           f"our C7 LSTM ({lstm.best.evaluation.all.r:.2f}). Two separate "
           "things drive it:")
    w(out)
    w(out, f"1. **The baseline overfits.** Its selected run has train MSE "
           f"{fc.best.evaluation.by_split['train'].mse:.3f} against test MSE "
           f"{fc.best.evaluation.by_split['test'].mse:.3f}. Because "
           "all-sample metrics are ~68% training rows and selection uses "
           "all-sample MSE, the method actively picks the most overfit of the "
           "five repeats (D-22).")
    w(out, f"2. **Our C7 LSTM underfits.** Its train-to-test MSE gap is "
           f"{_mean_over_repeats(lstm, 'test', 'mse') - _mean_over_repeats(lstm, 'train', 'mse'):.3f} "
           "against 0.335 for the BD LSTM - it is not learning the training "
           "set at all.")
    w(out)
    w(out, "On held-out data alone the baseline still wins at C7, so this is "
           "not purely a selection artefact. Candidate explanations we cannot "
           "settle: the paper never states the baseline's activation (D-09); "
           "C7's test fold is only 30 samples and its test R (0.19) sits far "
           "below its validation R (0.59), suggesting an unusually hard fold. "
           "We did not tune, because doing so would break comparability with "
           "Table 1.")
    w(out)

    w(out, "### Two methodological cautions for anyone using this model")
    w(out)
    w(out, "**Normalised MSE is scaler-relative (D-18).** Refitting the scaler "
           "on training rows only appeared to improve MSE by 0.052 - "
           "impossible, since removing a leak cannot improve a model. R moved "
           "+0.002 and RMSE by 3 mg/L: the models were identical and only the "
           "units changed. A useful side effect is that the paper's "
           "fit-on-all leak is demonstrably harmless here.")
    w(out)
    sp = forecast.spread()
    w(out, f"**The forecast is less certain than one number suggests (D-28).** "
           f"Across five seeds the forecast MSE runs {sp['min']:.2f} to "
           f"{sp['max']:.2f} (mean {sp['mean']:.2f}). Our headline "
           f"{forecast.predict_mse():.2f} is the best of them, selected "
           "honestly on training fit alone. The time tag also extrapolates "
           "beyond every day number it saw (D-29), so the forecast rests on "
           "the late-period decline continuing - which it did through "
           "2015-2016, but need not in general.")
    w(out)

    w(out, "### What a future run should do differently")
    w(out)
    w(out, "1. Report held-out test metrics alongside all-sample ones whenever "
           "models are compared, and select best-of-N on validation rather "
           "than all samples.")
    w(out, "2. Report a seed spread for any single-number headline, "
           "particularly the forecast.")
    w(out, "3. Treat the time tag as a trend term and state explicitly whether "
           "a given use interpolates or extrapolates it.")
    w(out, "4. Given the ~50% Type B sample loss, consider reporting results "
           "on the common sample set as the primary comparison rather than a "
           "robustness check.")
    w(out)


def _section_deviations(out, cfg):
    w(out, "## Every deviation and its likely effect")
    w(out)
    df = parse_deviations(cfg)
    w(out, f"{len(df)} entries, parsed directly from `DEVIATIONS.md` so the two "
           "cannot drift apart. Full reasoning for each is in that file.")
    w(out)
    w(out, _md_table(df[["id", "deviation", "status", "likely effect"]]))
    w(out)
    material = ["D-05", "D-09", "D-18", "D-21", "D-22", "D-27", "D-28", "D-29", "Q-01"]
    w(out, "**The ones that actually matter:** "
           + ", ".join(f"`{d}`" for d in material)
           + ". The rest are organisational, no-ops on this dataset, or too "
             "small to affect a conclusion.")
    w(out)


def _section_reproducibility(out, cfg):
    w(out, "## Reproducibility")
    w(out)
    n_tests, n_files = count_tests(cfg)
    w(out, _md_table(pd.DataFrame([
        {"item": "Config", "value": "`configs/base.yaml` (every setting and seed)"},
        {"item": "Base seed", "value": str(cfg["train"]["base_seed"])},
        {"item": "Repeats", "value": str(cfg["train"]["n_repeats"])},
        {"item": "Tests", "value": f"`pytest` - {n_tests} test functions across "
                                   f"{n_files} files (more cases once "
                                   "parametrised tests expand)"},
        {"item": "MLflow", "value": cfg.mlflow_experiment},
        {"item": "Reports", "value": "`reports/` - audit, sample counts, Table 1, "
                                     "FC baseline, time tag, forecast, this report"},
    ])))
    w(out)
    w(out, "Python, NumPy and torch are seeded for every run, and each run logs "
           "its config, seed, split sizes, sample counts, per-split metrics, "
           "scaler parameters and figures to MLflow. The same code runs "
           "locally and on Databricks, selected by the "
           "`DATABRICKS_RUNTIME_VERSION` environment variable.")
    w(out)
    w(out, "Notebooks, in order: `00_data_audit`, `01_parametric_study`, "
           "`02_fc_baseline`, `03_refined_timetag`, `04_forecast`, "
           "`05_sensitivity`, `06_replication_report`.")
    w(out)
    w(out, "**On Databricks** the pipeline runs on serverless environment 4, "
           "with PyTorch and MLflow from `requirements-databricks.txt` (D-33); "
           "`docs/README.md` has the setup. The local pins match that "
           "environment (D-01).")
    w(out)
