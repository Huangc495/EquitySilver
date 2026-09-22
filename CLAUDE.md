# CLAUDE.md: Replicating Ma et al. (2021) LSTM acidity–weather study

## Project goal

Replicate the following study in Python (PyTorch) on Azure Databricks:

> Ma, L., Huang, C., Liu, Z.-S., Morin, K.A., Aziz, M., Meints, C. (2021). *The correlation between drainage chemistry and weather for full-scale waste rock piles based on artificial neural network.* Journal of Contaminant Hydrology 239, 103793. https://doi.org/10.1016/j.jconhyd.2021.103793

The original study was built in MATLAB's Deep Learning Toolbox.

- The paper PDF lives at `docs/Ma_2021_JCH.pdf`. Consult it when this file is unclear.
- Where this file records a decision, this file wins.

### What the study does

An LSTM with one fully connected output layer maps weather sequences to drainage acidity at the Equity Silver mine's waste rock pile in BC.

- **Inputs:** daily total precipitation and mean temperature. These are windowed either as the 10 days ending on each acidity sample date (Type A) or as the 10 weeks ending on it (Type B).
- **Output:** acidity (mg/L as CaCO3) at two monitoring stations, BD and C7, modelled separately.
- **Experiments:**
  1. A parametric study over hidden sizes 5, 10 and 20 and input Types A and B.
  2. A comparison with a fully connected baseline.
  3. A "refined" model that adds a day-number time tag as an input feature.
  4. A 2-year forecast.
  5. A ±20% sensitivity test on the weather inputs.

---

## Working rules

1. **Work phase by phase.** At every **STOP** marker, write a short summary (what was done, key numbers, open questions) and wait for approval before continuing.
2. **Keep all settings in config.** Every hyperparameter and choice lives in `configs/base.yaml`. Never silently change a setting taken from the paper. Record every deviation from the paper, with its reason, in `DEVIATIONS.md`.
3. **Seed everything.** Seed Python, NumPy and torch, and log every seed. Runs must be reproducible.
4. **Keep notebooks thin.** Library code goes in `src/acidity_lstm/`. Notebooks only load config, call library functions, and display results.
5. **Run the same code in both places.** It must run locally (CPU, for development and tests) and on Databricks without edits. Detect Databricks via the `DATABRICKS_RUNTIME_VERSION` environment variable and choose paths from config.
6. **Test before experimenting.** `pytest` must pass before any experiment phase runs.
7. **Log every run to MLflow.** Log the config, seed, split sizes, sample counts, and the metrics for each split. Log figures and scaler parameters as artifacts.
8. **Never impute missing weather.** Samples with incomplete windows are dropped, as in the paper.
9. **Never commit data.** Keep `data/` in `.gitignore`.

---

## Data

A single Excel file (name to be confirmed in Phase 0) holds two kinds of data.

**Daily weather, 1997–2017.** This comes from Environment Canada, the paper's source (climate.weather.gc.ca).
- Expected fields include total precipitation (mm) and mean temperature (°C), possibly with min/max temperature, rain, snow, snow on ground, and data-quality flags.
- Only total precipitation and mean temperature are used as model inputs.
- The 1997 data exists only to supply lookback windows for early-1998 samples.

**Acidity at stations BD and C7.**
- Values are individual dated measurements, taken roughly twice a month, in mg/L as CaCO3.
- The paper uses 1998–2017, with 365 measurements at BD and 384 at C7.
- The paper notes that sampling is more frequent from March to July.

**Locations:**
- Local: `data/raw/<file>.xlsx`
- Databricks: `/Volumes/<catalog>/<schema>/raw/<file>.xlsx`

---

## Environment

**Databricks cluster.** Single node, CPU only, Databricks Runtime ML (LTS); PyTorch and MLflow come preinstalled. The model has about 500 parameters, so no GPU and no Spark parallelism are needed. Read the Excel file with pandas; install `openpyxl` if it's missing.

**Local environment.** A venv whose `requirements.txt` pins the same torch, numpy and pandas versions as the cluster. Get them from `pip freeze` on the cluster.

**Code workflow.** Develop locally, push to Git, and pull in a Databricks Git folder.
- Notebooks are `.py` files starting with `# Databricks notebook source`, so they diff cleanly in Git.
- Each notebook first adds `src/` to `sys.path`, or runs `%pip install -e` on the repo.

**MLflow.**
- Locally: tracking URI `file:./mlruns`.
- On Databricks: experiment `/Users/<you>/equity-silver-lstm`.

---

## Repo layout

```
equity-silver-lstm/
  CLAUDE.md
  DEVIATIONS.md
  requirements.txt
  configs/base.yaml
  docs/Ma_2021_JCH.pdf
  data/{raw,processed}/            # gitignored
  src/acidity_lstm/
    config.py        # load yaml, resolve local vs Databricks paths
    io.py            # read Excel, standardize columns
    preprocess.py    # calendar reindex, cleaning, duplicates
    windows.py       # Type A / Type B / time-tag sequence builders
    scaling.py       # z-score fit/transform/inverse
    models.py        # LSTMRegressor, FCBaseline, MATLAB-style init
    train.py         # SGDM loop, LR schedule, early stopping
    splits.py        # random and blocked split assignment
    evaluate.py      # MSE, R, RMSE (mg/L), figures
    experiments.py   # grids, repeats, MLflow logging
  notebooks/
    00_data_audit.py
    01_parametric_study.py         # Table 1, Figs 6-7
    02_fc_baseline.py
    03_refined_timetag.py          # Figs 9-10
    04_forecast.py                 # Fig 11
    05_sensitivity.py              # Fig 12
    06_replication_report.py
  tests/
    test_windows.py
    test_scaling.py
    test_models.py
  reports/                         # generated markdown + figures
```

---

## Config sketch (`configs/base.yaml`)

```yaml
paths:
  local:
    raw_excel: data/raw/equity_silver.xlsx
    processed_dir: data/processed
  databricks:
    raw_excel: /Volumes/<catalog>/<schema>/raw/equity_silver.xlsx
    processed_dir: /Volumes/<catalog>/<schema>/processed
  reports_dir: reports
mlflow:
  local_experiment: equity-silver-lstm
  databricks_experiment: /Users/<you>/equity-silver-lstm
data:
  weather_start: 1997-01-01
  acidity_start: 1998-01-01
  acidity_end: 2017-12-31
  stations: [BD, C7]
  same_day_duplicates: mean        # mean | keep
windows:
  n_steps: 10
  types: [A, B]
  common_sample_set: false         # true = run Type A only on samples that survive Type B
  time_tag: per_step               # per_step | constant
  time_tag_origin: 1998-01-01      # day 1
scaling:
  fit_on: all                      # all (paper) | train
  ddof: 0                          # population std
model:
  hidden_sizes: [5, 10, 20]
  output_activation: linear        # linear | sigmoid (Eq. 7 as printed)
  init: matlab                     # matlab | torch_default
  freeze_bias_hh: false
  fc_baseline_hidden: 10
  fc_baseline_activation: tanh
train:
  lr: 0.005
  momentum: 0.9
  weight_decay: 1.0e-4             # MATLAB default L2Regularization
  lr_drop_period: 125
  lr_drop_factor: 0.2
  batch_size: 20
  drop_last: true                  # MATLAB discards the incomplete final mini-batch
  shuffle: once                    # once (MATLAB default) | every_epoch
  max_epochs: 200
  patience: 6
  restore_best: false              # MATLAB returns the last-iteration network
  n_repeats: 5
  base_seed: 42
split:
  method: random                   # random (paper) | blocked
  fractions: [0.70, 0.15, 0.15]
  resplit_each_repeat: false
forecast:
  station: BD
  train_years: [1999, 2014]
  predict_years: [2015, 2016]
  fractions: [0.80, 0.20]
  scaler_fit_on: train_period
sensitivity:
  factors: [0.8, 1.2]
  variables: [precip, tmean]
```

---

## Phase 0: Data audit

Write `notebooks/00_data_audit.py` and produce `reports/00_data_audit.md` plus figures. The audit must cover:

1. **File structure.** Sheet names, columns, data types, units, and date ranges for each sheet.
2. **Weather continuity.** Confirm the weather data is continuous daily.
   - List missing days and gap runs (start, end, length), separately for precipitation and mean temperature.
   - Check whether a station name or ID column exists, and whether the station changes over time.
3. **Weather quality flags.** If Environment Canada flag columns are present, report their values. The default plan is to treat "M" (missing) as NaN and "T" (trace) as 0 mm; confirm this with the user.
4. **Acidity dates.** Confirm every acidity value has a full date.
5. **Acidity counts.** Give counts per station per year, and totals for 1998–2017 compared with the paper's counts (BD 365, C7 384). Flag any large mismatch.
6. **Acidity data quality.**
   - Same-day duplicate measurements per station.
   - Non-numeric or censored entries.
   - The value range. Paper Fig. 4 shows values mostly between 5,000 and 30,000 mg/L, with occasional peaks near 40,000.
7. **Seasonality of sampling.** How samples are distributed across months.
8. **Surviving samples.** A preliminary count, per station, of samples that survive the complete-window rule for Type A (10 days) and for Type B (70 days).
9. **Plots for comparison with paper Fig. 4.** The acidity time series for BD and C7, and the weather series with gaps visible.

**STOP.** Report the audit and wait for approval.

---

## Phase 1: Cleaning (`io.py`, `preprocess.py`)

1. **Standardize columns.**
   - Weather: `date, precip_mm, tmean_c`.
   - Acidity, in long format: `station, date, acidity_mgL`.
2. **Reindex the weather** onto a continuous daily calendar from `weather_start` to 2017-12-31. Gaps stay NaN; do not interpolate or fill them.
3. **Clean the acidity data.**
   - Coerce values to numeric and log every coerced entry.
   - Keep only dates from 1998-01-01 to 2017-12-31.
   - Handle same-day duplicates as set by `same_day_duplicates` (default: average them) and log the count.
4. **Save the outputs** as parquet in `processed_dir`.

---

## Phase 2: Building samples (`windows.py`)

### Defining a sample

- One sample is one acidity measurement at one station on date **d**.
- BD and C7 get separate sample sets, because they are usually not sampled on the same days.

### Windows

Every window is anchored on the sample's own date **d**, and each window includes day d itself (paper Fig. 5 shows steps −9…0).

**Type A.** The daily values for days d−9 through d.

**Type B.** Week k (k = 0…9) spans days [d−7k−6, d−7k].
- Weekly precipitation is the sum of the 7 daily values. Weekly temperature is the mean of the 7 daily means.
- Compute these from the daily series for each sample. Do **not** use a calendar-week table: each sample's weeks start on a different weekday.

**Tensor layout.** Steps are ordered oldest to newest, giving shape (N, 10, 2). The last step is the current day or week.

**Missing data.** Drop a sample if any day in its window is NaN. That means 10 days must be complete for Type A and 70 days for Type B.
- Log the surviving counts per station and type.
- Type A and Type B will end up with different sample sets. `common_sample_set: true` runs Type A only on the samples that survive Type B.

**Time tag (refined model).**
- Day number = (date − 1998-01-01) in days + 1, so 1998-01-01 is day 1.
- `per_step` (default): each step carries its own day number. For Type A that is each day's number; for Type B it is the number of the week's last day (d−7k).
- `constant`: every step carries day d's number.
- The tag becomes a third feature, giving shape (N, 10, 3).

### Implementation

- Store the weather as NumPy arrays indexed by the integer day offset from `weather_start`.
- Extract windows by vectorized index arithmetic, with no per-sample pandas filtering.
- Keep a metadata DataFrame row-aligned with the tensor: `station, date, acidity_mgL, day_number, split`. The time-series figures are drawn from it.

### Required tests (`tests/test_windows.py`)

1. **Window boundaries.** A sample dated 1998-03-15 should give:
   - Type A: 1998-03-06 to 1998-03-15.
   - Type B: week 0 = 1998-03-09 to 03-15, and week 9 = 1998-01-05 to 01-11, for 70 days in total.
2. **1997 lookback.** A sample dated 1998-01-20 draws its Type B window partly from 1997; the earliest day is 1997-11-12.
3. **Missing-day handling.** A NaN placed 30 days before the sample drops it from Type B but keeps it in Type A.
4. **Aggregation.** Weekly sums and means match a hand calculation on a small synthetic series.
5. **Ordering.** Oldest to newest along the step axis.
6. **Time tag.** The `per_step` and `constant` variants produce the expected day numbers.

**STOP.** Report that the tests pass and give the final sample counts per station and type.

---

## Phase 3: Normalization (`scaling.py`)

- **Method.** Z-score (paper Eqs. 8–10) with population standard deviation (`ddof=0`). Eq. 9 is misprinted without the square, but it is clearly meant to be the standard deviation.
- **Inputs.** Compute one mean and std per feature (precipitation, temperature, and the time tag if used), pooled over all time steps of all samples in the fitting set.
- **Output.** Compute one mean and std for acidity per station.
- **Fitting set.**
  - `fit_on: all` (paper default) fits on all samples of the scenario, which leaks into the test set.
  - `fit_on: train` fits on training samples only. Report both.
- **Scope.** Fit a separate scaler for each station × input type × time-tag setting.
- **Logging and inverse transform.** Log scaler parameters with each run. Provide an inverse transform for computing metrics in mg/L.

---

## Phase 4: Models (`models.py`)

### `LSTMRegressor(n_features, hidden_size)`

- One `nn.LSTM(batch_first=True)` layer. Take the hidden state at the last step, then apply `nn.Linear(hidden_size, 1)`.
- **Output activation:** linear by default. Paper Eq. 7 shows a sigmoid, but a sigmoid cannot produce z-scores outside (0, 1), and MATLAB's `fullyConnectedLayer` + `regressionLayer` setup is linear. Keep sigmoid as a config option.

**MATLAB-style initialization (`init: matlab`).**
- `weight_ih`: Xavier/Glorot uniform.
- `weight_hh`: orthogonal.
- Forget-gate bias = 1 and all other biases = 0. PyTorch's gate order is i, f, g, o, so the forget slice is `[H:2H]`. Put the 1 in `bias_ih` and zero `bias_hh`.
- `Linear`: Glorot weights, zero bias.

**Parameter count check** for H=10 with 2 inputs:
- MATLAB count: 531 (LSTM 4·10·12 + 40 = 520, plus FC 11).
- PyTorch count: 571, because PyTorch keeps a second bias vector (`bias_hh`).
- `freeze_bias_hh: true` pins it at zero, making the effective model match MATLAB.

### `FCBaseline`

After Ma et al. (2020a): flatten the Type B input (10 × 2 = 20 values), then `Linear(20, 10)`, tanh, `Linear(10, 1)`. The paper doesn't give this model's activation; tanh is an assumption, and it is MATLAB's default for this kind of network. Record it in `DEVIATIONS.md`.

**Tests (`tests/test_models.py`).** Output shapes, the parameter count above, the forget-gate bias value, and that the network overfits a tiny batch.

---

## Phase 5: Training (`train.py`, `splits.py`)

| Setting | Paper | Implementation |
|---|---|---|
| Optimizer | SGDM, momentum 0.9 | `torch.optim.SGD(lr=0.005, momentum=0.9, weight_decay=1e-4)` |
| Learning-rate drop | ×0.2 after 125 epochs | `StepLR(step_size=125, gamma=0.2)` |
| Mini-batch | 20 | `batch_size=20`, `drop_last=True` (MATLAB behaviour) |
| Shuffle | not stated | `once` (MATLAB default); `every_epoch` optional |
| Max epochs | 200 | 200 |
| Early stopping | validation stops improving for 6 epochs | validate every epoch; stop after 6 epochs without a strictly lower validation MSE |
| Returned network | not stated | final network (`restore_best: false`, MATLAB-like) |
| Loss | MSE | `nn.MSELoss` on normalized values |
| Repeats | 5, keep the lowest MSE | 5 re-initializations; log all 5 plus mean ± SD |

In practice, early stopping will almost always trigger before epoch 125, so the learning-rate drop rarely takes effect.

### Splits

**Assign once, before windowing.** For each station, assign every acidity measurement to train, validation or test once, using a seeded random draw. Samples keep that assignment after windowing, so all input types and hidden sizes for a station share the same split.

**`method: random`** (paper). A 70/15/15 split by measurement.

**`method: blocked`** (robustness check). Hold out whole calendar years: about 14/3/3 of the 20 years, chosen by seed.

This check matters because samples are about 2 weeks apart. Consecutive Type B windows therefore share about 55 of their 70 days, and a random split puts near-identical inputs in both training and test. That flatters the validation and test scores, mostly for Type B.

**`resplit_each_repeat: false`** (default). The split stays fixed within a scenario, and only the initialization varies across the 5 repeats. This follows the paper's stated reason for repeating: random initialization.

### Metrics (`evaluate.py`)

- MSE in normalized units: all samples, train, validation and test.
- Pearson R (paper Eq. 12) for the same subsets.
- RMSE in mg/L after the inverse transform.
- **Best-of-5 selection** uses MSE over *all* samples, which matches how Table 1 reports "lowest MSE."

---

## Phase 6: Parametric study (Table 1, Figs 6–7)

**Grid:** station {BD, C7} × input {A, B} × hidden size {5, 10, 20} × 5 repeats, for 60 runs. That takes minutes on CPU, so run them sequentially.

**Outputs:**
- `reports/table1.md`: best-of-5 MSE next to the paper's values, a mean ± SD table, and MSE and R for each split.
- **Fig. 6:** normalized measured vs calculated acidity, with a fit line and R, for BD and C7 (Type B, H=10, all samples).
- **Fig. 7:** BD measured vs calculated acidity over time (Type B, H=10).
- The same grid under `split: blocked`, and optionally with `common_sample_set: true`.

**STOP.** Report the results against the paper's targets.

---

## Phase 7: Fully connected baseline

Train the FC baseline on the Type B input, using the same splits, training settings and repeats. Compare R for BD and C7 with the LSTM's (Type B, H=10).

---

## Phase 8: Refined model with the time tag (Figs 9–10)

**Setup:** station BD, Type B, H=10, with the time tag as a third feature. All other settings as in Phase 6.

**Outputs:**
- **Fig. 9:** original vs refined scatter, both with fit lines and R.
- **Fig. 10:** refined measured vs calculated acidity over time.
- Run both time-tag variants (`per_step` and `constant`) and report both.

---

## Phase 9: Forecast (Fig. 11)

1. **Train** the refined model (BD, Type B, H=10) on samples dated 1999–2014 only, with a random 80/20 train/validation split and no test set. 1998 and 2017 are excluded, as in the paper.
2. **Fit the scaler** on 1999–2014 only. This is the honest choice for a forecast; document it.
3. **Predict** every 2015–2016 BD sample using the actual weather. Windows for early 2015 may reach back into late-2014 weather, which is fine.
4. **Select** the best of 5 by train + validation MSE.
5. **Report:**
   - MSE on train + validation (paper: 0.28) and on the prediction period (paper: 0.23).
   - RMSE in mg/L.
   - The spread of prediction-period MSE across the 5 seeds.
   - **Fig. 11:** measured vs predicted acidity for 2015–2016.
6. **Note in the report** that the time tag extrapolates beyond its training range here.

---

## Phase 10: Sensitivity (Fig. 12)

1. Use the Phase 9 model and its scaler.
2. Multiply the raw daily values for 2015-01-01 to 2016-12-31 **only** by 0.8 and by 1.2, one variable at a time. That is precipitation in mm and mean temperature in °C. Scaling °C amplifies both summer highs and winter lows, which matches the paper's idea of larger "temperature fluctuation."
3. Rebuild the windows, apply the unchanged scaler, and predict.
4. **Outputs:**
   - **Fig. 12:** five curves (real weather, precipitation ×0.8, precipitation ×1.2, temperature ×0.8, temperature ×1.2).
   - A table of the mean change in predicted acidity for each scenario.
5. **Expected directions (paper):**
   - More precipitation → lower acidity.
   - A larger temperature swing → higher acidity.
   - Temperature is the more sensitive input.

---

## Phase 11: Replication report

Write `reports/replication_report.md`, containing:
- A table comparing the paper's result with ours for every target below.
- Every deviation from `DEVIATIONS.md` with its likely effect.
- The blocked-split robustness results.
- A short discussion of which qualitative findings replicated.

**STOP.**

---

## Paper targets

All MSEs are in normalized units.

| Experiment | Paper result |
|---|---|
| Table 1, BD, H = 5 / 10 / 20 | Type A 0.79 / 0.76 / 0.76; Type B 0.59 / 0.54 / 0.54 |
| Table 1, C7, H = 5 / 10 / 20 | Type A 0.85 / 0.83 / 0.81; Type B 0.77 / 0.74 / 0.73 |
| Fig. 6 (Type B, H=10) | R = 0.70 (BD), 0.51 (C7) |
| FC baseline (Type B, 10 neurons) | R = 0.64 (BD), 0.42 (C7) |
| Refined model with time tag (BD, Type B, H=10) | R = 0.86, MSE = 0.26 (vs 0.70 and 0.54 without the tag) |
| Forecast (BD) | MSE 0.28 on 1999–2014 train + validation; 0.23 on 2015–2016 |
| Sensitivity | more precipitation → lower acidity; larger temperature swing → higher acidity; temperature is the more sensitive input |

**Don't chase exact numbers.** With only about 300 samples per station and best-of-5 selection, exact matches are not expected. Do not tune beyond the paper's settings to hit these targets.

The real test of the replication is the qualitative findings:
1. Type B beats Type A.
2. BD fits better than C7.
3. The LSTM beats the FC baseline.
4. The time tag improves the fit substantially.
5. The forecast is reasonable.
6. The sensitivity directions hold.

---

## Paper ambiguities and default decisions

Each of these is a config option. Log the chosen value in every run.

| Ambiguity | Default | Alternative |
|---|---|---|
| Output activation (Eq. 7 shows a sigmoid) | linear | sigmoid |
| Normalization fitting set (paper fits on all data) | all samples | training samples only |
| Std convention (Eq. 9 misprinted) | population, ddof = 0 | — |
| Window alignment (Fig. 5 includes the current day/week) | includes day d | — |
| Meaning of "lowest MSE" in best-of-5 | MSE over all samples | — |
| Repeats | split fixed, initialization varies | re-split each repeat |
| Early stopping | return the final network (MATLAB behaviour) | restore the best network |
| Shuffling and final mini-batch | shuffle once, drop last partial batch (MATLAB defaults) | shuffle every epoch |
| Time tag | per step | constant (sample's day number) |
| Same-day duplicate acidity | average | keep all |
| Type A vs Type B sample sets | each type keeps its own survivors | common sample set |
| FC baseline activation | tanh (assumption) | — |
