# Replication report

**Ma, L., Huang, C., Liu, Z.-S., Morin, K.A., Aziz, M., Meints, C. (2021).** *The correlation between drainage chemistry and weather for full-scale waste rock piles based on artificial neural network.* Journal of Contaminant Hydrology 239, 103793.

A PyTorch replication of a study originally built in MATLAB's Deep Learning Toolbox, run on the Equity Silver mine's own weather record and drainage chemistry.

## Verdict

**The study replicates.** All 12 Table 1 cells land within **0.045 MSE** of the published values (mean 0.020), and **five of the six qualitative findings hold**. The sixth holds at one station and fails at the other.

Nothing was tuned beyond the paper's stated settings. Where the paper is ambiguous, the choice was made in `configs/base.yaml`, recorded in `DEVIATIONS.md`, and in the two cases that mattered (`fit_on`, `time_tag`) both options were run and shown to make no difference.

The three results worth carrying forward are not the headline numbers:

1. **Normalised MSE is not comparable across different scaler fits** (D-18). Overlooking this produces a spurious 0.052 MSE "improvement" that scale-free metrics show to be nothing.
2. **All-sample metrics and best-of-5 selection both reward overfitting** (D-22). This is the paper's own methodology, and it inflated our C7 baseline's headline R from an honest 0.39 to 0.78.
3. **The forecast's headline is the best of a wide spread** (D-28). A typical run scores about double the reported value.

## Data

The strongest single piece of evidence that the pipeline matches the paper came before any modelling. The acidity workbook holds 367 BD and 385 C7 measurements for 1998-2017; dropping the five blank `ACIDITY` cells leaves **exactly 365 and 384**, the paper's own counts. The weather is the paper's own source (EQUITY SILVER, Climate ID 1072692), continuous daily 1997-2017 with no missing rows.

13.7% of weather days are missing a value, and the paper states it dropped incomplete-window samples rather than gap-filling. Applying the same rule leaves:

| station | measurements | Type A | Type B |
|---------|--------------|--------|--------|
| BD      | 365          | 274    | 185    |
| C7      | 384          | 278    | 186    |

The paper never reports its surviving counts, so this cannot be checked against it. With ~185 Type B samples every metric is noisy: run-to-run SD (0.030 MSE) exceeds our mean gap to the paper (0.020).

## Every paper target

| target                                      | paper              | ours                    | worst gap | match             |
|---------------------------------------------|--------------------|-------------------------|-----------|-------------------|
| Table 1, BD, Type A (H = 5/10/20)           | 0.79 / 0.76 / 0.76 | 0.78 / 0.76 / 0.78      | 0.017     | yes               |
| Table 1, BD, Type B (H = 5/10/20)           | 0.59 / 0.54 / 0.54 | 0.61 / 0.56 / 0.55      | 0.020     | yes               |
| Table 1, C7, Type A (H = 5/10/20)           | 0.85 / 0.83 / 0.81 | 0.81 / 0.83 / 0.84      | 0.045     | yes               |
| Table 1, C7, Type B (H = 5/10/20)           | 0.77 / 0.74 / 0.73 | 0.74 / 0.71 / 0.70      | 0.035     | yes               |
| Fig. 6 R (Type B, H = 10), BD / C7          | 0.70 / 0.51        | 0.67 / 0.56             | 0.05      | yes               |
| FC baseline R (Type B, 10 neurons), BD / C7 | 0.64 / 0.42        | 0.54 / 0.78             | 0.36      | BD yes, C7 no     |
| Refined model (BD, Type B, H = 10), R       | 0.86 (from 0.70)   | 0.83 (from 0.67)        | 0.03      | yes               |
| Refined model, MSE                          | 0.26 (from 0.54)   | 0.32 (from 0.56)        | 0.06      | yes               |
| Forecast, train + validation MSE            | 0.28               | 0.25                    | 0.03      | yes               |
| Forecast, prediction-period MSE             | 0.23               | 0.26 (5-seed mean 0.47) | 0.03      | yes, but see D-28 |
| Sensitivity directions                      | 3 of 3             | 3 of 3                  | -         | yes               |

All MSEs are in normalised units. Forecast MSEs use a training-period scaler and are **not** comparable with Table 1's (D-18, D-27).

## The six qualitative findings

CLAUDE.md names these as the real test of the replication.

| # | finding                                     | verdict     | evidence                                                                                    |
|---|---------------------------------------------|-------------|---------------------------------------------------------------------------------------------|
| 1 | Type B beats Type A                         | REPLICATED  | mean best MSE 0.643 (B) vs 0.798 (A); B wins 6/6 pairs, and still wins on identical samples |
| 2 | BD fits better than C7                      | REPLICATED  | mean Type B MSE 0.572 (BD) vs 0.714 (C7)                                                    |
| 3 | LSTM beats the FC baseline                  | PARTIAL     | BD yes (held-out too); C7 no (FC wins on held-out data)                                     |
| 4 | The time tag improves the fit substantially | REPLICATED  | R 0.67 -> 0.83 (+0.16), the same size gain as the paper's +0.16                             |
| 5 | The forecast is reasonable                  | REPLICATED* | forecast MSE 0.26 vs paper 0.23; *but the 5-seed mean is 0.47 (D-28)                        |
| 6 | The sensitivity directions hold             | REPLICATED  | 3/3 directions, including temperature as the more sensitive input                           |

**Score: 5 replicated, 1 partial.**

## Blocked-split robustness

Samples sit about two weeks apart, so consecutive Type B windows share roughly 55 of their 70 days. A random split can therefore place near-identical inputs in both training and test, flattering the scores. Holding out whole calendar years removes that leak.

| input type | mean MSE change under blocked split |
|------------|-------------------------------------|
| A          | -0.008                              |
| B          | 0.02                                |

**The leak is real but modest.** Type B degrades by +0.020 MSE while Type A is essentially unchanged (-0.008) - the direction the overlap argument predicts, since the longer window shares more days between neighbours. It is nowhere near large enough to overturn finding 1: Type B still beats Type A comfortably under blocking.

### The time tag under blocking

| model              | test R (random split) | test R (blocked split) |
|--------------------|-----------------------|------------------------|
| Original           | 0.454                 | 0.228                  |
| Refined (time tag) | 0.528                 | 0.322                  |

The tag's advantage on held-out samples is +0.074 R under the random split and +0.094 under blocking, so it is not purely an interpolatable trend. Both models lose absolute accuracy under blocking, as expected.

**Caveat:** blocked folds are drawn from the 14 years that contain Type B survivors, giving ~10/2/2 years rather than 14/3/3 (D-13), so these numbers are noisier than the random-split ones. And because most held-out years sit inside the training range, this still tests interpolation; true extrapolation appears only in the Phase 9 forecast (D-29).

## Discussion

### What replicated cleanly

Findings 1, 2, 4 and 6 replicate without qualification, and the Table 1 numbers match closely enough that the remaining gaps (mean 0.020 MSE) are smaller than our own run-to-run spread (0.030). Given about 300 measurements per station, best-of-5 selection, and a MATLAB-to-PyTorch port, this is about as close as the design allows.

The time-tag result is the most striking: the R gain of +0.16 matches the paper's exactly, from a starting point 0.03 below theirs.

### Where it did not: finding 3 at C7

Our C7 FC baseline reaches all-sample R 0.78 against the paper's 0.42, beating our C7 LSTM (0.56). Two separate things drive it:

1. **The baseline overfits.** Its selected run has train MSE 0.286 against test MSE 0.861. Because all-sample metrics are ~68% training rows and selection uses all-sample MSE, the method actively picks the most overfit of the five repeats (D-22).
2. **Our C7 LSTM underfits.** Its train-to-test MSE gap is 0.062 against 0.335 for the BD LSTM - it is not learning the training set at all.

On held-out data alone the baseline still wins at C7, so this is not purely a selection artefact. Candidate explanations we cannot settle: the paper never states the baseline's activation (D-09); C7's test fold is only 30 samples and its test R (0.19) sits far below its validation R (0.59), suggesting an unusually hard fold. We did not tune, because doing so would break comparability with Table 1.

### Two methodological cautions for anyone using this model

**Normalised MSE is scaler-relative (D-18).** Refitting the scaler on training rows only appeared to improve MSE by 0.052 - impossible, since removing a leak cannot improve a model. R moved +0.002 and RMSE by 3 mg/L: the models were identical and only the units changed. A useful side effect is that the paper's fit-on-all leak is demonstrably harmless here.

**The forecast is less certain than one number suggests (D-28).** Across five seeds the forecast MSE runs 0.26 to 0.57 (mean 0.47). Our headline 0.26 is the best of them, selected honestly on training fit alone. The time tag also extrapolates beyond every day number it saw (D-29), so the forecast rests on the late-period decline continuing - which it did through 2015-2016, but need not in general.

### What a future run should do differently

1. Report held-out test metrics alongside all-sample ones whenever models are compared, and select best-of-N on validation rather than all samples.
2. Report a seed spread for any single-number headline, particularly the forecast.
3. Treat the time tag as a trend term and state explicitly whether a given use interpolates or extrapolates it.
4. Given the ~50% Type B sample loss, consider reporting results on the common sample set as the primary comparison rather than a robustness check.

## Every deviation and its likely effect

34 entries, parsed directly from `DEVIATIONS.md` so the two cannot drift apart. Full reasoning for each is in that file.

| id   | deviation                                                           | status                        | likely effect                                                                                                                |
|------|---------------------------------------------------------------------|-------------------------------|------------------------------------------------------------------------------------------------------------------------------|
| D-00 | Databricks workspace settings                                       | partially resolved            | None on results. Deployment configuration; the catalog and schema are still outstanding.                                     |
| D-01 | Dependency pins are not yet confirmed against the cluster           | open                          | None on results. Environment reproducibility only; confirm the pins against the cluster.                                     |
| D-02 | Raw data layout differs from the config sketch                      | active                        | None. Same data source as the paper (Equity Silver, Climate ID 1072692).                                                     |
| D-03 | `audit.py` added to the library                                     | active                        | None. Code organisation only.                                                                                                |
| D-04 | Markdown tables are rendered by hand, not by `tabulate`             | active                        | None. Report formatting only.                                                                                                |
| D-05 | Blank acidity cells are dropped                                     | active, validated             | Decisive and validated. Reproduces the paper's BD 365 / C7 384 exactly.                                                      |
| D-06 | Environment Canada flag rules are no-ops                            | no-op                         | None on this dataset; both flag rules are no-ops here.                                                                       |
| D-07 | `A`, `C`, `E` and `F` flagged weather values are kept               | active                        | Small. Retains ~256 lower-quality precipitation days that would otherwise widen the gaps and shrink Type B further.          |
| D-08 | Same-day duplicate policy never fires                               | no-op                         | None. No same-day duplicates exist after the 1998-2017 filter.                                                               |
| D-09 | FC baseline activation is assumed to be tanh                        | active, unresolved            | Possibly material for finding 3. An unstated activation could explain part of the C7 FC discrepancy.                         |
| D-10 | Output layer is linear, not the sigmoid printed in Eq. 7            | active                        | Large if wrong, but almost certainly right: a sigmoid output cannot emit z-scores, so the paper's MSEs would be unreachable. |
| D-11 | PyTorch carries a second bias vector                                | active                        | Small. 40 extra free parameters at H=10 (571 vs MATLAB's 531).                                                               |
| D-12 | Early stopping returns the final network                            | active                        | Small. Returning the final rather than the best network slightly worsens reported metrics.                                   |
| D-13 | Blocked splits are drawn from eligible years only                   | active                        | Moderate on the blocked-split numbers only. 14 eligible years give ~10/2/2 folds rather than 14/3/3, so they are noisier.    |
| D-14 | Realised split proportions drift from 70/15/15                      | active, expected              | None, and unavoidable. A direct consequence of sharing one split across input types, as CLAUDE.md specifies.                 |
| D-15 | `resplit_each_repeat` needs a rebuild function                      | active                        | None by default (`resplit_each_repeat: false`).                                                                              |
| D-16 | A single short batch is kept when a split is smaller than the batch | active                        | None on the main grid; relevant only to small subsets.                                                                       |
| D-17 | Orthogonal initialisation is applied per gate                       | active                        | None detectable. The closest PyTorch equivalent of MATLAB's orthogonal recurrent initialisation.                             |
| D-18 | Normalised MSE is not comparable across different scaler fits       | active, important             | Important for interpretation, none on the models. Normalised MSEs are only comparable within a shared scaler.                |
| D-19 | Four variants are run, not the two `CLAUDE.md` lists                | active                        | None on the headline. Adds robustness evidence beyond the paper.                                                             |
| D-20 | Figures are logged on a summary MLflow run                          | active                        | None. Logging only.                                                                                                          |
| D-21 | Qualitative finding 3 only partially replicates                     | open discrepancy              | This IS a result, not a cause. Finding 3 fails at C7; see the discussion.                                                    |
| D-22 | All-sample metrics and best-of-5 selection both reward overfitting  | active, methodological        | Material for model comparison. Inflates any overfitting model's headline score, including the paper's own.                   |
| D-23 | The `time_tag` ambiguity is immaterial                              | resolved                      | None. The two time-tag settings are numerically equivalent here.                                                             |
| D-24 | A blocked-split check was added to Phase 8                          | active, extension             | None on the headline. Adds evidence that the tag's gain is not purely interpolation.                                         |
| D-25 | Forecast splits are assigned after windowing                        | active                        | Negligible. Gives an exact 80/20; the time-based forecast period is unaffected.                                              |
| D-26 | The forecast period reuses the `test` split label                   | active, internal              | None. Naming only.                                                                                                           |
| D-27 | Forecast MSEs are not comparable with Table 1's                     | active, consequence of D-18   | Important for interpretation. Phase 9 MSEs must not be compared with Table 1's.                                              |
| D-28 | The headline forecast MSE is the best of a wide spread              | open caveat                   | Material for reading the forecast. The headline is the best of five; a typical run is about twice the paper's value.         |
| D-29 | The time tag genuinely extrapolates in Phase 9                      | active, quantified            | Material for trusting the forecast. The tag is unconstrained outside its training range.                                     |
| D-30 | POSIX-absolute config paths were re-rooted on Windows               | fixed                         | None on any reported result; every number came from the local path set. Fixed, with a regression test.                       |
| Q-01 | Type B loses about half the samples                                 | open, quantified in Phase 6   | Moderate. ~185 Type B samples make every metric noisy; run-to-run SD exceeds our mean gap to the paper.                      |
| Q-02 | Type B is a strict subset of Type A                                 | resolved in Phase 6           | None, once tested. Type B still beats Type A on identical samples.                                                           |
| Q-03 | Whole-year blocked splits are not directly possible                 | resolved in Phase 5, see D-13 | Resolved in D-13.                                                                                                            |

**The ones that actually matter:** `D-05`, `D-09`, `D-18`, `D-21`, `D-22`, `D-27`, `D-28`, `D-29`, `Q-01`. The rest are organisational, no-ops on this dataset, or too small to affect a conclusion.

## Reproducibility

| item      | value                                                                                     |
|-----------|-------------------------------------------------------------------------------------------|
| Config    | `configs/base.yaml` (every setting and seed)                                              |
| Base seed | 42                                                                                        |
| Repeats   | 5                                                                                         |
| Tests     | `pytest` - 219 test functions across 11 files (more cases once parametrised tests expand) |
| MLflow    | equity-silver-lstm                                                                        |
| Reports   | `reports/` - audit, sample counts, Table 1, FC baseline, time tag, forecast, this report  |

Python, NumPy and torch are seeded for every run, and each run logs its config, seed, split sizes, sample counts, per-split metrics, scaler parameters and figures to MLflow. The same code runs locally and on Databricks, selected by the `DATABRICKS_RUNTIME_VERSION` environment variable.

Notebooks, in order: `00_data_audit`, `01_parametric_study`, `02_fc_baseline`, `03_refined_timetag`, `04_forecast`, `05_sensitivity`, `06_replication_report`.

**Outstanding before a Databricks run:** the dependency pins are not yet confirmed against the cluster (D-01), and `configs/base.yaml` still carries `<catalog>`, `<schema>` and `<you>` placeholders for the Volumes paths and the MLflow experiment.

