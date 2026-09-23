# Deviations from Ma et al. (2021)

Every departure from the paper, or from the plan in `CLAUDE.md`, with its
reason. Ambiguities that `CLAUDE.md` already resolved are listed under
"Resolved ambiguities" rather than as deviations.

Status key: **active** = in force; **no-op** = implemented but has no effect on
this dataset; **open** = needs a decision from the user.

---

## Environment and repo

### D-00 Databricks workspace settings — *resolved*
Workspace: **Equity-Silver-Databricks-MLOps** (Azure Databricks).

MLflow experiment: `/Users/${DATABRICKS_USERNAME}/equity-silver-lstm`. The
workspace username is **not** stored in this repo, which is public. `${VAR}`
references in config values are expanded from the environment at load time by
`config.expand_env`, which raises a pointed error naming the variable if it is
unset — deliberately not `os.path.expandvars`, which would leave a literal
`${...}` in the path and fail confusingly later.

Set it once on the cluster: **Compute > Edit > Advanced options > Spark >
Environment variables**, `DATABRICKS_USERNAME=you@example.com`. The same
mechanism works for `paths.databricks` if you would rather not commit the
catalog and schema either.

Unity Catalog: catalog **`equity_silver_databricks_mlops`**, schema
**`default`**, with `raw` and `processed` as two volumes. Volume paths are
`/Volumes/<catalog>/<schema>/<volume>/<path>`, so those last segments are
volume names rather than folders. `docs/README.md` carries the `CREATE VOLUME`
statements and the upload step; `data/` stays out of the repository.

**A correction worth recording:** the schema was first given as
`equity_silver_databricks_mlops.information_schema`. That could not have
worked for two reasons — `information_schema` is read-only system metadata
that Unity Catalog auto-creates in every catalog and which cannot hold
volumes, and the value was catalog-qualified where the path needs the bare
schema name. Both would only have surfaced as a runtime failure on the
cluster, so `tests/test_config.py` now asserts the cluster paths are
well-formed volume paths: four segments, a bare schema name, and never
`information_schema`.

**Volumes created and raw data uploaded** by the user. Not verifiable from
the development machine (no Databricks CLI or credentials there), so
`notebooks/00a_cluster_preflight.py` checks it on the cluster instead — see
D-31.

**Confirmed 2026-09-22 (M0).** The preflight passed on the workspace: all 21
weather CSVs and the acidity workbook are in the `raw` volume, both volumes
are writable, and the MLflow experiment resolves with no environment
variable set.

### D-01 Dependency pins are confirmed against the platform — *resolved*
`CLAUDE.md` says to pin `requirements.txt` from `pip freeze` on the Databricks
cluster. No cluster access was available, so the pins target **Databricks
Runtime 16.4 LTS ML** (Python 3.12, numpy 1.26.4, pandas 2.2.3, torch 2.6.0,
mlflow 2.21.3).

**Effect:** none on results if the cluster matches; a pandas/numpy major
mismatch could change behaviour. **Action:** confirm with `pip freeze` on the
cluster and re-pin.

**Resolved 2026-09-22 (M0).** There is no Runtime ML cluster: the workspace
is serverless-only (D-33). The preflight's version report from serverless
environment 4 is now what `requirements.txt` pins. The one major change was
numpy 1.26.4 → 2.1.3. Rerunning every notebook locally on the new pins
reproduced all 66 result rows (Table 1, FC baseline, time tag,
sensitivity) **bit for bit**; only wall-clock seconds and matplotlib's
rendering of the figures differ. One cosmetic difference surfaced: numpy 2
ordered equal-length weather gaps differently in the audit's "longest runs"
table, so ties there are now broken by start date.

### D-02 Raw data layout differs from the config sketch — *active*
`CLAUDE.md` describes "a single Excel file" holding both weather and acidity.
The delivered data is:

- **Weather:** 21 Environment Canada CSVs, one per year, 1997-2017
  (`en_climate_daily_BC_1072692_<year>_P1D.csv`).
- **Acidity:** `2017 ARD Chemisty_Clean.xlsx`, two sheets
  (`Bessmemer Dump` -> BD, `C-7` -> C7).

Config therefore has `weather_glob` and `acidity_excel` instead of a single
`raw_excel`. **Effect:** none on results; the data is the paper's own source
(Equity Silver, Climate ID 1072692).

### D-03 `audit.py` added to the library — *active*
The repo layout in `CLAUDE.md` does not list an audit module, but rule 4 says
notebooks stay thin. Phase 0 logic lives in `src/acidity_lstm/audit.py` and
`notebooks/00_data_audit.py` only calls it. The Phase 2 counts report
(`run_sample_count_report` -> `reports/02_sample_counts.md`) is emitted from the
same module and notebook, since `CLAUDE.md` gives Phases 1-2 no notebook of
their own. **Effect:** none on results.

### D-04 Markdown tables are rendered by hand, not by `tabulate` — *active*
`DataFrame.to_markdown` requires `tabulate`, which is not guaranteed on a
Databricks ML runtime. `audit._md_table` formats tables directly.
**Effect:** none on results; avoids a fragile dependency.

---

## Data handling

### D-05 Blank acidity cells are dropped — *active, validated*
Five `ACIDITY` cells are blank (BD 2000-09-20 and 2007-08-02; C7 1993-02-04,
1993-02-25 and 2007-08-02). They are coerced to NaN and dropped.

**This is validated, not assumed.** Dropping them yields exactly **BD 365** and
**C7 384** for 1998-2017 — the paper's reported counts. Keeping them would give
367 and 385. **Effect:** reproduces the paper's sample counts exactly.

### D-06 Environment Canada flag rules are no-ops — *no-op*
Config sets `T` (trace) -> 0 mm and `M` (missing) -> NaN. On this dataset every
`T` row already records 0.0 mm and every `M` row is already blank, so neither
rule changes a value. They are kept as safeguards.

**The substantive point:** most missing weather carries *no* flag — 761 blank
unflagged precipitation days vs 20 `M` flags, and 784 vs 254 for mean
temperature. All blanks are treated as missing and never imputed.

### D-07 `A`, `C`, `E` and `F` flagged weather values are kept — *active*
`A` (accumulated), `C` (amount uncertain), `E` (estimated) and `F` (accumulated
and estimated) rows carry values and are used as recorded. The paper does not
mention flag filtering. Discarding them would drop a further ~256 precipitation
days and widen the gaps, shrinking the already-thin Type B sample set.
**Effect:** a small amount of lower-quality precipitation data is retained.

### D-08 Same-day duplicate policy never fires — *no-op*
`same_day_duplicates: mean` is configured, but there are **zero** same-day
duplicate measurements at either station after the 1998-2017 filter. This
ambiguity does not affect the replication.

---

## Modelling (planned; recorded here in advance)

### D-09 FC baseline activation is assumed to be tanh — *active, unresolved*
The paper does not state the fully connected baseline's activation.
`fc_baseline_activation: tanh` follows MATLAB's default for this network type.

**Phase 7 makes this matter more than expected.** Finding 3 replicates at BD
but inverts at C7, where our FC baseline reaches R = 0.78 over all samples
against the paper's 0.42. Some of that gap could come from a different
activation. It cannot be resolved from the paper, and tuning it to match would
violate the no-tuning rule.

### D-10 Output layer is linear, not the sigmoid printed in Eq. 7 — *active*
Eq. 7 as printed shows a sigmoid output, but the target is a z-score, which a
sigmoid cannot produce outside (0, 1). MATLAB's
`fullyConnectedLayer` + `regressionLayer` pairing is linear. `sigmoid` is kept
as a config option. **Effect:** a sigmoid output would make the reported MSEs
unreachable, so this is almost certainly what the paper actually ran.

### D-11 PyTorch carries a second bias vector — *active*
PyTorch's LSTM has both `bias_ih` and `bias_hh`, so H=10 with 2 inputs gives
571 parameters against MATLAB's 531. `freeze_bias_hh: true` pins `bias_hh` at
zero to match MATLAB's effective parameter count. Default is `false`.
**Effect:** 40 extra free parameters at H=10; small, but recorded.

### D-12 Early stopping returns the final network — *active*
The paper does not say whether the returned network is the best or the last.
`restore_best: false` follows MATLAB's behaviour. **Effect:** test metrics may
be marginally worse than a best-checkpoint policy would give.

---

## Open questions for the user

### Q-01 Type B loses about half the samples — *open, quantified in Phase 6*
The complete-window rule leaves **BD 274 / C7 278** samples for Type A but only
**BD 185 / C7 186** for Type B. The paper confirms it used the same rule
("Those acidity measurements without a complete time series input data will be
neglected") but **never reports its surviving counts**, so this cannot be
checked against it.

With ~185 Type B samples a 15% test split is ~28 points. Expect noisy
run-to-run scatter and do not read too much into small MSE differences.

**Phase 6 confirms the concern quantitatively:** run-to-run SD averages 0.030
MSE, larger than the 0.020 mean gap to the paper, and test-split R swings from
0.14 to 0.53 across cells. Best-of-5 MSE is nonetheless within 0.045 of the
paper in all 12 cells.

### Q-02 Type B is a strict subset of Type A — *resolved in Phase 6*
Every Type B survivor is also a Type A survivor, so the headline "Type B beats
Type A" comparison is confounded: it changes the sample set as well as the
input representation.

`common_sample_set: true` restricts Type A to the 185 / 186 Type B survivors,
so both types are compared on identical samples.

**Phase 6 result: the confound does not explain the finding.** On identical
samples Type B still beats Type A at every station and hidden size, by 0.155 to
0.215 MSE at BD and 0.075 to 0.114 at C7. "Type B beats Type A" is about the
input representation, not the sample set.

### Q-03 Whole-year blocked splits are not directly possible — *resolved in Phase 5, see D-13*


---

## Phases 3-5 (scaling, models, training)

### D-13 Blocked splits are drawn from eligible years only — *active*
Resolves Q-03. BD Type B has no surviving sample in 1998, 2007, 2010, 2011,
2012 or 2017, so partitioning all 20 calendar years blindly could put only
empty years in the validation or test fold.

`splits.blocked_split` accepts `eligible_years`; when given, only years holding
at least one usable sample are partitioned, and the remaining years are
labelled `train`, where they contribute nothing because they hold no surviving
samples. `splits.eligible_years_from` computes the list from the most
restrictive scenarios (Type B).

The eligible pool for Type B is 14 years, not 20, so the fractions give roughly
**10 / 2 / 2** rather than the 14 / 3 / 3 sketched in `CLAUDE.md`.
**Effect:** blocked-split results rest on fewer held-out years than planned;
this is a property of the data, not a choice.

### D-14 Realised split proportions drift from 70/15/15 — *active, expected*
`CLAUDE.md` requires splits to be assigned **once per station before
windowing**, so every input type shares one split. Because the complete-window
rule then drops samples unevenly, the realised proportions after windowing are
not exactly 70/15/15. For BD Type B at seed 42 they come out **130 / 26 / 29**
(70.3 / 14.1 / 15.7 %).

This is a direct consequence of the design `CLAUDE.md` specifies, not a bug.
Realised counts are reported with every run rather than corrected, since
rebalancing would break the shared split.

### D-15 `resplit_each_repeat` needs a rebuild function — *active*
Re-splitting per repeat requires the raw measurements, which the training layer
does not hold. `train.train_repeats` therefore takes an optional `rebuild_fn`
and **raises** if `resplit_each_repeat` is true without one, rather than
silently ignoring the setting. **Effect:** none by default
(`resplit_each_repeat: false`).

### D-16 A single short batch is kept when a split is smaller than the batch — *active*
MATLAB drops the incomplete final mini-batch (`drop_last: true`). Applied
literally, a training split with fewer than 20 samples would yield no batches
at all and train on nothing. `train._iter_batches` falls back to one short
batch in that case. **Effect:** none on the main grid, where every training
split has 126-189 samples; it matters only for small sensitivity or forecast
subsets.

### D-17 Orthogonal initialisation is applied per gate — *active*
MATLAB initialises the recurrent weights of `lstmLayer` orthogonally. PyTorch
stacks the four gates into one `(4H, H)` matrix, which cannot be orthogonal as
a whole. `models.matlab_init` orthogonalises each gate's `(H, H)` block
separately, which is the closest equivalent. **Effect:** matches MATLAB's
intent; verified by test.

---

## Phases 1-2 outcomes (no new deviations)

Phase 1 cleaning and Phase 2 windowing introduced no further departures from the
paper. The results they lock in:

| Quantity | Value |
|---|---|
| Weather calendar | 7,670 continuous daily rows, 1997-01-01 to 2017-12-31, no duplicates |
| Missing weather values | precip 781, tmean 1,038 — preserved as NaN, never imputed |
| Acidity after cleaning | **BD 365, C7 384** (exactly Ma et al.) |
| Type A survivors | BD 274, C7 278 |
| Type B survivors | BD 185, C7 186 |
| Common sample set | BD 185, C7 186 for both types |
| Time tag | adds a third feature; does not change which samples survive |
| Forecast split (BD, Type B) | 161 samples 1999-2014, 24 samples 2015-2016 |

All of these are locked by tests (`tests/test_preprocess.py`,
`tests/test_windows.py`), so a future change that breaks the match with the
paper's counts will fail the suite rather than pass silently.


---

## Phase 6 outcomes

### D-18 Normalised MSE is not comparable across different scaler fits — *active, important*
Discovered while comparing the `fit_train` variant. Normalised MSE is expressed
in units of the fitted standard deviation, so refitting the scaler changes the
units and the numbers are no longer on a common scale.

Fitting the scaler on training rows only appeared to **improve** MSE by 0.052,
which cannot be right — removing a leak cannot improve a model. Scale-free
metrics show the models are in fact indistinguishable: Pearson R moves by
+0.002 and RMSE by −3 mg/L against values of 2,000–5,000 mg/L.

**Consequences:**
1. The paper's `fit_on: all` leak is **harmless** on this dataset, which
   resolves that ambiguity. The paper's setting stays the default.
2. MSEs may only be compared between runs sharing a normalisation. Every
   Table 1 comparison does, so they remain valid. Later phases must not compare
   MSE across differently-scaled runs — this matters for Phase 9, where the
   forecast scaler is fitted on 1999–2014 only.

### D-19 Four variants are run, not the two `CLAUDE.md` lists — *active*
`CLAUDE.md` Phase 6 asks for the grid under `split: blocked` and optionally
`common_sample_set: true`. Phase 3 separately asks to "report both" `fit_on`
settings. All are run as named variants (`paper`, `blocked`, `common`,
`fit_train`), 4 × 12 × 5 = 240 runs in about 50 seconds on CPU.
**Effect:** none on the headline result; adds the robustness evidence.

### D-20 Figures are logged on a summary MLflow run — *active*
Figures are produced after the grid finishes, so they cannot belong to any one
scenario run. `experiments.log_study_artifacts` logs them, `table1.md` and
`table1_raw.csv` on a `phase6-summary` run. Per-scenario runs additionally log
`scaler.json` as an artifact. **Effect:** satisfies rule 7; none on results.

### Phase 6 headline results

| Check | Result |
|---|---|
| Table 1 agreement | all 12 cells within **0.045** MSE (mean 0.020) |
| Fig. 6 R, BD | 0.67 (paper 0.70) |
| Fig. 6 R, C7 | 0.56 (paper 0.51) |
| Finding 1: Type B beats Type A | **replicated**, 6/6 pairs, and on identical samples |
| Finding 2: BD fits better than C7 | **replicated**, 0.572 vs 0.714 mean Type B MSE |
| Blocked split | +0.020 MSE for Type B, −0.008 for Type A |
| Runs logged to MLflow | 289 (48 scenarios + 240 repeats + 1 summary) |


---

## Phases 7-8 outcomes

### D-21 Qualitative finding 3 only partially replicates — *open discrepancy*
"The LSTM beats the FC baseline" holds at **BD** on both the paper's metric
(all-sample R 0.67 vs 0.54) and on held-out test data (mean test R 0.454 vs
0.318). It **fails at C7**, where the FC baseline wins on both (all-sample R
0.78 vs 0.56; test R 0.362 vs 0.187).

Diagnosis, from the per-repeat metrics:

1. **The FC baseline overfits badly at C7.** Its selected run has train MSE
   0.286 against test MSE 0.861 — a gap of 0.575 — and trained 59 epochs
   against the LSTM's 20. Its headline R of 0.78 is largely memorised training
   data.
2. **Our C7 LSTM underfits.** Train MSE 0.771 against test MSE 0.834, a gap of
   just 0.062 versus 0.335 for the BD LSTM. It is not learning the training set.
3. Even on held-out data the FC baseline still wins at C7, so this is not
   purely a selection artefact.

Caveats: the C7 test fold holds 30 samples (test-R SD 0.050 across repeats),
and C7 test R (0.19) is far below C7 validation R (0.59), suggesting an
unusually hard fold. Not patched by tuning; carried to Phase 11.

### D-22 All-sample metrics and best-of-5 selection both reward overfitting — *active, methodological*
Discovered in Phase 7. All-sample MSE and R are dominated by the ~68% of rows
that are training samples, and best-of-5 selection uses all-sample MSE — the
paper's stated rule and CLAUDE.md's chosen reading. Among five repeats it
therefore prefers the one that overfits hardest.

This is a property of the paper's own methodology, not of this implementation,
so the rule is kept as specified. Reports now give held-out test metrics
alongside the all-sample ones wherever a model comparison is drawn.

### D-23 The `time_tag` ambiguity is immaterial — *resolved*
`per_step` and `constant` give the same model to three decimal places
(MSE 0.318 vs 0.318, R 0.826 vs 0.826). The reason is quantified:

- A Type B window spans 63 days, so the two encodings differ by at most 63 in
  day number.
- The tag is z-scored over 1998-2017, where its standard deviation is ~1,963
  days, so they differ by at most **3.2% of one standard deviation**.
- Measured directly, the within-window spread of the normalised tag is ~1% of
  the between-sample spread.

The tag acts almost purely as a per-sample index. `per_step` stays the default;
either choice gives the same answer.

### D-24 A blocked-split check was added to Phase 8 — *active, extension*
`CLAUDE.md` Phase 8 does not ask for one, but the time tag is a monotonic index
and could simply be fitting the long-term trend. The blocked split holds out
whole years, so the model must predict unseen dates.

**Result: the tag's advantage survives.** Its gain in held-out test R is +0.074
under the random split and +0.094 under the blocked split. Both models lose
absolute accuracy under blocking (original test R 0.454 -> 0.228; refined 0.528
-> 0.322), but the tag's relative contribution holds.

Note this still tests *interpolation to unseen dates*, since most held-out
years sit inside the training range. **True extrapolation is only tested in
Phase 9**, where the tag must run beyond every day number it saw.

### Phase 7-8 headline results

| Check | Result |
|---|---|
| Finding 3: LSTM beats FC baseline | **partially replicated** — holds at BD, fails at C7 (D-21) |
| BD: LSTM vs FC, all-sample R | 0.67 vs 0.54 (paper 0.70 vs 0.64) |
| C7: LSTM vs FC, all-sample R | 0.56 vs 0.78 (paper 0.51 vs 0.42) |
| Finding 4: time tag improves the fit | **replicated** |
| BD refined, R | 0.67 -> **0.83** (paper 0.70 -> 0.86) |
| BD refined, MSE | 0.56 -> **0.32** (paper 0.54 -> 0.26) |
| Size of the R gain | **+0.16, identical to the paper's +0.16** |
| Time-tag mode | immaterial (D-23) |


---

## Phases 9-10 outcomes

### D-25 Forecast splits are assigned after windowing — *active*
Phase 5 assigns splits on measurements *before* windowing so every input type
shares one split. The forecast has only one input type and a **time-based**
held-out period, so that reason does not apply. Splits are assigned on the 161
surviving training-period samples instead, giving an exact 80/20 (129/32).
Assigning beforehand would have given an uneven ratio after dropping.
**Effect:** negligible; the forecast period is unaffected either way.

### D-26 The forecast period reuses the `test` split label — *active, internal*
`CLAUDE.md` specifies no test set inside the training period. The 2015-2016
forecast samples are labelled `test` so the existing evaluation, logging and
figure code reports them without special-casing. There is no held-out test
*within* 1999-2014; `test` means "forecast period" throughout Phase 9.
**Effect:** none on results; naming only.

### D-27 Forecast MSEs are not comparable with Table 1's — *active, consequence of D-18*
The forecast scaler is fitted on 1999-2014 only (161 samples), so its
normalised units differ from the Phase 6-8 scaler's. Phase 9 MSEs may be
compared with **the paper's forecast figures** (same construction) and with
each other, but **not** with Table 1 or the Phase 8 refined MSE of 0.32.

### D-28 The headline forecast MSE is the best of a wide spread — *open caveat*
Selection is on train+validation MSE only, never on the forecast period, so
there is no leakage. But the five seeds give forecast MSEs of 0.265, 0.432,
0.526, 0.560 and 0.569 — **mean 0.470, SD 0.127**.

The selected run (seed 45) happens to be both the best training fit *and* the
best forecast, so the headline **0.265** against the paper's 0.23 is luck, not
method. A typical run is about **0.47**, roughly double the paper's figure.

With 24 forecast samples this spread is unsurprising. The paper reports a
single value without a spread, so we cannot tell where its 0.23 sits in its own
distribution. **The qualitative finding still holds** — the forecast is
reasonable — but the point estimate should not be read as a tight match.

### D-29 The time tag genuinely extrapolates in Phase 9 — *active, quantified*
Every forecast sample carries a day number beyond the training range:

| period | day numbers | normalised |
|---|---|---|
| training 1999-2014 | 400 to 6,189 | −1.34 to +2.53 |
| forecast 2015-2016 | 6,218 to 6,846 | +2.55 to +2.97 |

The network has no evidence about that range and can only continue the trend it
fitted. Phase 8's blocked-split check held out years *inside* the training
range, so it tested interpolation; this is the first true extrapolation test,
and the wide seed spread (D-28) is consistent with an unconstrained feature.

The forecast works here partly because BD acidity declined steadily through the
late training period and kept declining through 2015-2016. A trend-following
tag extrapolates well while the trend persists; it would be unsafe where the
trend changes.

### Phase 9-10 headline results

| Check | Result |
|---|---|
| Forecast, train+validation MSE | **0.253** (paper 0.28) |
| Forecast period MSE, selected run | **0.265** (paper 0.23) |
| Forecast period MSE, 5-seed mean ± SD | **0.470 ± 0.127** (range 0.265–0.569) |
| Forecast RMSE | **1,473 mg/L**, ~13% of the mean measured value |
| Finding 5: forecast is reasonable | **replicated**, with the D-28 caveat |
| More precipitation → lower acidity | **yes** (−122 mg/L at ×1.2, +225 at ×0.8) |
| Larger temperature swing → higher acidity | **yes** (+472 mg/L at ×1.2, −408 at ×0.8) |
| Temperature is the more sensitive input | **yes** (8.0% total swing vs 3.2%) |
| Finding 6: sensitivity directions hold | **replicated, 3/3** |


---

## Phase 11 outcomes

No new deviations. The replication report (`reports/replication_report.md`)
parses this file directly for its deviation table, and
`report_phase11.LIKELY_EFFECT` must carry an effect assessment for every entry
here — adding a deviation without one raises, so the two cannot drift apart.

### Final scorecard

| Paper target | Result |
|---|---|
| Table 1, all 12 cells | within **0.045 MSE** (mean 0.020) |
| Fig. 6 R (BD / C7) | 0.67 / 0.56 vs 0.70 / 0.51 |
| FC baseline R (BD / C7) | 0.54 / 0.78 vs 0.64 / 0.42 — **C7 fails** |
| Refined model R | 0.83 vs 0.86 (gain +0.16, identical to the paper's) |
| Refined model MSE | 0.32 vs 0.26 |
| Forecast train+val / predict MSE | 0.25 / 0.26 vs 0.28 / 0.23 (see D-28) |
| Sensitivity directions | 3 of 3 |

| Qualitative finding | Verdict |
|---|---|
| 1. Type B beats Type A | **replicated** (6/6 pairs, and on identical samples) |
| 2. BD fits better than C7 | **replicated** |
| 3. LSTM beats the FC baseline | **partial** — BD yes, C7 no (D-21) |
| 4. Time tag improves the fit | **replicated** |
| 5. Forecast is reasonable | **replicated**, with the D-28 spread caveat |
| 6. Sensitivity directions hold | **replicated**, 3/3 |

**5 replicated, 1 partial.**


---

## Post-replication fixes

### D-30 POSIX-absolute config paths were re-rooted on Windows — *fixed*
Found while verifying the Databricks path set after filling in the workspace
username.

`Config.resolve` used `Path.is_absolute()` to decide whether to join a
configured path to the repo root. On Windows a bare leading slash is **not**
absolute, so `/Volumes/<catalog>/<schema>/raw` silently became
`C:\Volumes\<catalog>\<schema>
aw` — re-rooted onto the repo's drive.

The cluster itself was never affected (Linux treats the path as absolute), so
this would only ever have surfaced as a confusing failure when simulating or
testing the Databricks path set from a Windows machine. `config._resolve_against`
now treats a leading "/" as absolute on every platform.

**Effect:** none on any result in this replication — every reported number came
from the local path set. Covered by `tests/test_config.py`, including a
regression test for the re-rooting and one that fails if any `<placeholder>`
remains in the MLflow experiment path.


### D-31 A preflight notebook was added — *active, extension*
Not in the `CLAUDE.md` layout. Added because three things can only be
confirmed on the cluster, and each otherwise fails part-way through a notebook
with a misleading error: `DATABRICKS_USERNAME` unset, raw data missing or
partially uploaded to the Volume, and packages disagreeing with
`requirements.txt`.

`notebooks/00a_cluster_preflight.py` and `preflight.py` check all three,
change nothing, and are safe to re-run. Failures are hard (missing data,
unresolvable config); version disagreements are warnings, since on a cluster
the runtime's own version is the one to trust.

`preflight.environment_report()` prints the installed versions as a
requirements block, which is how **D-01** gets closed from the cluster itself.

**Effect:** none on results. 17 tests, mostly driving the failure paths — a
preflight that cannot fail is worthless.

---

## MLOps stage

Entries from here on belong to the MLOps stage (`MLOPS.md`). Its decisions
are numbered D1–D6 there; the entries below are departures from the
replication's own plan made along the way.

### D-32 Cluster reports go to a volume, not the repo — *active*
`CLAUDE.md` puts generated reports in `reports/` at the repo root, and until
now that held on every platform. On Databricks the repo is a **Git folder**.
A run that writes `reports/` there leaves tracked files modified in the
workspace, and the next pull of `main` can then fail with a conflict.

`reports_dir` now lives in each path set, like the data paths:

| Where | `reports_dir` |
|---|---|
| local | `reports` (unchanged) |
| Databricks | `/Volumes/equity_silver_databricks_mlops/default/processed/reports` |

`Config.reports_dir` resolves it the same way as every other path.
`test_config.py` checks that the cluster path is a well-formed volume path
and lies outside the repo, and that the local path is still `reports/`.

This is the minimum change that lets phase M0 run from the Git folder. In
production, outputs move to Unity Catalog tables (M4), and this volume
directory is an interim location.

**Effect:** none on results. Reports from a cluster run land in the volume,
so comparing them with the committed local reports means reading them from
there.

### D-33 Serverless compute replaces the Runtime ML cluster — *active*
`CLAUDE.md` specifies a single-node CPU cluster on Databricks Runtime ML
(LTS), and `HANDOVER.md` gave a cluster spec for it. **This workspace cannot
run one.** Every classic cluster request, all-purpose or job, is rejected
with *"Current organization … does not have any associated worker
environments"*: the workspace has no classic compute plane. The Databricks
CLI retries that error silently, so a `clusters create` call simply hangs;
`--debug` shows it. The user chose serverless (`MLOPS.md`, D7).

Serverless lacks PyTorch and MLflow, which the previous session took to
mean it could not run the pipeline. A probe run on 2026-09-22 showed that
an environment supplies them:

| Piece | Where |
|---|---|
| Environment version | `compute.serverless_environment_version: "4"` in `configs/base.yaml`; Python 3.12.3 |
| Packages serverless lacks | `requirements-databricks.txt`: CPU-only torch 2.6.0 wheel, mlflow 2.21.3, openpyxl 3.1.2 |
| Everything else | provided by environment 4 |

The CPU wheel is named explicitly because the PyPI build of torch for Linux
pulls about 2.5 GB of CUDA libraries that serverless would download on
every run.

**Local pins follow the platform.** `CLAUDE.md` asks for the local venv to
match the cluster's torch, numpy and pandas. Environment 4 has **numpy
2.1.3**, where the replication ran on 1.26.4, so `requirements.txt` is
re-pinned to environment 4: numpy 2.1.3, pyarrow 19.0.1, matplotlib 3.10.0,
scipy 1.15.1, PyYAML 6.0.2 and pytest 8.3.5, with pandas 2.2.3, torch 2.6.0,
mlflow 2.21.3 and openpyxl 3.1.2 unchanged. This closes D-01.

**Preflight.** The "Runtime ML" check is now a **compute** check. An ML
runtime passes. Serverless passes, with a warning if its environment
version differs from config, because torch and mlflow are verified package
by package anyway. A standard runtime still fails. On serverless, a missing
torch or mlflow names `requirements-databricks.txt` rather than telling the
user to create a cluster they cannot have.

**Found along the way:** the handover's cluster spec set no
`data_security_mode`. A classic cluster created through the API without
one cannot read Unity Catalog volumes; ML runtimes need `SINGLE_USER`.
It is moot here, but it matters for anyone reusing that spec elsewhere.

**Effect:** none on the model or the method. Whether the numpy 1 → 2
change moves any number is checked by rerunning the pipeline locally
against the committed reports (M0).

### D-34 CI runs the unit tests only; the data-dependent tests are marked — *active*
`CLAUDE.md` rule 6 says `pytest` must pass before any experiment runs. The
suite was written assuming the real data sits in `data/raw`. The data is
gitignored, so a CI runner would never have it, and 71 of the 270 tests
failed or errored without it.

Those 71 now carry the `integration` marker, registered in
`pyproject.toml`. `test_phases78.py` is marked as a whole module; the rest
are marked test by test, so the unit tests beside them still run in CI.
The marking was checked both ways:

| Check | Result |
|---|---|
| `-m "not integration"` with the data hidden | 199 passed, 0 errors |
| `-m integration` with the data present | 71 passed: exactly the set that broke without data |
| full suite with the data present | all pass |

`tests/conftest.py` skips integration tests with a stated reason when the
data is missing, rather than letting 66 fixtures error one by one.
`ACIDITY_REQUIRE_DATA=1` turns that into a hard failure, for places where
the data is supposed to be present.

**Rule 6 still holds where the data lives.** CI (`azure-pipelines.yml`)
proves the code; the full suite, integration tests included, must still
pass locally or on Databricks before an experiment runs.

**Effect:** none on results. The tests that lock the paper's sample
counts, BD 365 and C7 384, are integration tests, so CI cannot catch a
regression in them. Only a run with the data can.

### D-35 The first registered models are the replication's own cells — *active*
Phase M2 registers one model per station in Unity Catalog, with two
versions told apart by alias (`MLOPS.md`, D3). The **champion** is Type B,
H=10, untagged. The **challenger** is the same model with the per-step time
tag, and it runs in shadow. Both are trained by the replication's own code
(`experiments.run_cell`) under the paper recipe: random 70/15/15 split,
scaler fitted on all samples, best of 5 by all-sample MSE. The champion is
therefore the same model as the matching Table 1 / Fig. 6 cell, and the
challenger the same as the Phase 8 model.

This is a deliberate starting point, not the production recipe. It
inherits D-22: best-of-5 on all-sample MSE favours the most overfit
repeat. Phase M3's promotion gate re-judges models on held-out years, in
R and RMSE in mg/L, across several seeds.

**What serving adds.** The registered pyfunc takes raw daily weather and
returns mg/L. It builds its inputs with `windows.window_features`, the
function now shared with `build_samples`: the window code was factored out
of `build_samples` unchanged, and every existing test still passes. It
also carries its own scaler, and ships the package as `code_paths`. A test
loads the registered model and checks that, fed the real weather, it
returns the training code's predictions at every sample date. A missing
day blanks every window that covers it, and nothing is imputed.

`evaluate.predict` moved to `models.predict`, and `evaluate` re-exports
it, so serving needs only torch and not the plotting stack.

**Effect:** none on any replication result. The champion's quoted
performance carries D-22's optimism until M3 re-judges it.

### D-36 Production models use a backtested, 5-seed recipe — *active*
From M3, the models that reach `@champion` and `@challenger` are no longer
made the paper's way (`MLOPS.md` D9, D10):

| | Paper recipe (replication, M2) | Production recipe (M3) |
|---|---|---|
| Split | random 70/15/15 by measurement | none: refit on every year |
| Seeds | best of 5 by all-sample MSE | the average of all 5 |
| Epochs | early stopping on validation rows | fixed: the backtest's median `epochs_run` |
| Scaler | fitted on all samples | fitted on all samples, which are all training rows here |
| How it is judged | all-sample MSE (D-22) | out-of-fold R and RMSE in mg/L from blocked 5-fold CV by year |

Inside the backtest, each fold's scaler is fitted on its training rows
only. When the paper recipe is backtested as an incumbent, its best-of-5
is chosen on training and validation rows, never on the held-out fold.

**First comparison on the real data** (local, 5 folds x 5 seeds), against
the paper recipe:

| Slot | Paper: out-of-fold R / RMSE | Production: R / RMSE | Gate |
|---|---|---|---|
| BD champion | 0.538 / 2487 mg/L | 0.542 / 2448 | promoted |
| BD challenger | 0.691 / 2152 | 0.694 / 2101 | promoted |
| C7 champion | 0.419 / 5268 | 0.436 / 5213 | promoted |
| C7 challenger | 0.586 / 4806 | 0.558 / 4805 | rejected: R fell |

These are the first held-out-year figures for either recipe. BD's
champion reaches R 0.54 out of fold, against 0.67 over all samples: the
held-out years give the honest number. The ensemble's gains are small, R
+0.004 to +0.017, well within the noise. With zero tolerance the gate
decides on differences that small, and whether it should is an open
threshold question for the user.

**Effect:** none on the replication. Production models are judged on
held-out years, so their quoted performance is lower than the
replication's and more honest.
