# Handover: from replication to Azure MLOps

**To the incoming session.** The replication of Ma et al. (2021) is finished.
Your job is the next stage: run the same study on Azure as a production ML
system, with CI/CD for ML, experiment tracking, a model registry, and model
monitoring.

This file tells you what exists, what state the Azure deployment is in, which
decisions belong to the user, and which traps have already been hit once.
Everything here was checked against the repository on 2026-09-22 at commit
`4698d2f`.

---

## 0. Read this section first

**What exists.** A tested Python/PyTorch package (`src/acidity_lstm/`, 17
modules), 8 Databricks notebooks, 264 passing tests, and a replication report.
It runs locally and was built to run on Databricks without edits.

**What has never happened.** The pipeline has **never run end to end on
Databricks**. Every reported number came from local runs on the development
machine. Getting one clean Databricks run is the first milestone, before any
MLOps work.

**Three gaps block MLOps directly**, and all three were verified, not assumed:

1. **No trained model is ever saved.** MLflow receives params, metrics,
   `scaler.json` and figures, but there is no `mlflow.pytorch.log_model` or
   pyfunc anywhere in `src/` or `notebooks/`. There is nothing to register
   yet.
2. **No packaging.** There is no `pyproject.toml` or `setup.py`. All 8
   notebooks reach the library by inserting `src/` into `sys.path`.
3. **No CI, and the tests assume local data.** There is no `.github/`
   directory. With `data/raw/` hidden, **71 of 264 tests fail** (5 failures,
   66 errors) and 193 pass. `data/` is gitignored, so a GitHub runner would be
   red from the first push.

**Before building anything, settle the six decisions in section 3 with the
user.**

**Documents to read, in order:**

| File | Why |
|---|---|
| `CLAUDE.md` | Loaded into your session automatically. Its **working rules** still apply; its Phases 0–11 are complete. |
| `HANDOVER.md` | This file. |
| `DEVIATIONS.md` | 35 entries: decisions D-00 to D-31 and open questions Q-01 to Q-03. Continue from **D-32**. |
| `reports/replication_report.md` | Results, with every paper target compared. |
| `docs/README.md` | Databricks setup: cluster, volumes, upload, Git folder, notebook order. |
| `docs/architecture.html` | Interactive map of which module calls which. Open it in a browser. |

---

## 1. The replication result

**It replicates.** All 12 Table 1 cells land within **0.045 MSE** of the
paper (mean gap 0.020), and **5 of 6 qualitative findings hold**. The sixth
holds at one station and fails at the other.

| Paper target | Paper | Ours |
|---|---|---|
| Table 1, 12 cells (best-of-5 MSE) | — | all within 0.045 |
| Fig. 6 R, Type B H=10 (BD / C7) | 0.70 / 0.51 | 0.67 / 0.56 |
| FC baseline R (BD / C7) | 0.64 / 0.42 | 0.54 / **0.78** (overfit) |
| Refined model with time tag, R | 0.70 → 0.86 | 0.67 → 0.83 |
| Refined model, MSE | 0.54 → 0.26 | 0.56 → 0.32 |
| Forecast MSE, train+val / 2015–16 | 0.28 / 0.23 | 0.25 / 0.26 |
| Forecast MSE, mean ± SD over 5 seeds | not reported | **0.47 ± 0.13** |
| Sensitivity directions | 3 of 3 | 3 of 3 |

| Finding | Verdict |
|---|---|
| 1. Type B beats Type A | replicated, and still holds on identical samples |
| 2. BD fits better than C7 | replicated |
| 3. The LSTM beats the FC baseline | **partial**: holds at BD, fails at C7 (D-21) |
| 4. The time tag improves the fit | replicated; the +0.16 R gain matches the paper exactly |
| 5. The forecast is reasonable | replicated, but see D-28 |
| 6. Sensitivity directions hold | replicated |

**Data facts the tests lock in:** BD 365 and C7 384 measurements, exactly the
paper's counts. Surviving samples: Type A 274/278, Type B 185/186. The
forecast has 161 training-period samples and 24 forecast samples. The weather
record has 7,670 days, 13.7% of them missing a value, and nothing is ever
imputed.

---

## 2. Azure and Databricks: current state

| Item | State |
|---|---|
| Workspace | **Equity-Silver-Databricks-MLOps** (Azure Databricks) |
| Unity Catalog | catalog `equity_silver_databricks_mlops`, schema `default` |
| Volumes | `raw` and `processed`, created by the user |
| Raw data (22 files) | uploaded to `raw`. **Inferred, not seen:** the user's preflight run failed only on `torch` and `mlflow`, so the data checks passed |
| Repository in a Git folder | yes; the user ran `00a_cluster_preflight` from it |
| Compute | **serverless, which cannot run this.** A classic ML cluster spec was given (below); not yet confirmed created |
| Preflight fully green | **no** |
| Notebooks 00–06 on Databricks | **never run** |
| MLflow runs in the workspace | none. About 750 runs sit only in the local `./mlruns`, which is gitignored |
| `requirements.txt` confirmed against the cluster (D-01) | **no**; pinned by inference to DBR 16.4 LTS ML |
| Workspace access from the dev machine | **none**: no Databricks CLI and no credentials |

### The cluster spec already given to the user

Serverless compute has no PyTorch, no runtime selector and no cluster
environment variables. The pipeline needs a classic single-node cluster on an
**ML** runtime:

```json
{
  "cluster_name": "equity-silver-ml",
  "spark_version": "16.4.x-cpu-ml-scala2.12",
  "node_type_id": "Standard_DS3_v2",
  "num_workers": 0,
  "spark_conf": {
    "spark.databricks.cluster.profile": "singleNode",
    "spark.master": "local[*]"
  },
  "custom_tags": { "ResourceClass": "SingleNode" },
  "autotermination_minutes": 30
}
```

The user creates it via **Compute > Create compute > JSON**. `node_type_id`
may need changing by region. The same spec is the natural job-cluster
definition for an Asset Bundle later.

### How the configuration resolves

- `DATABRICKS_RUNTIME_VERSION` selects the local or Databricks path set and the
  MLflow experiment. The same code runs in both places.
- The MLflow experiment is `/Users/${DATABRICKS_USERNAME}/equity-silver-lstm`.
  `config.expand_env` resolves `DATABRICKS_USERNAME` from the environment, and
  otherwise from the signed-in user via Spark's `current_user()`. **The
  username is deliberately not stored in the repository.**
- Cluster paths are `/Volumes/equity_silver_databricks_mlops/default/{raw,processed}/…`.

---

## 3. Decisions that belong to the user

Do not start building until these are settled. Each changes what gets built.
Recommendations are offered, not decided.

**D1. Which platform.** The request said "Azure MLOps", which could mean Azure
Databricks' native tooling or Azure Machine Learning.
*Recommend Databricks-native:* the workspace already exists, is literally named
for MLOps, and holds the data in Unity Catalog. Azure ML would add a second
workspace, a second registry and a data-movement problem.

**D2. What "production" means here.** Acidity is sampled about twice a month
and weather arrives daily, so nobody needs a sub-second answer.
*Recommend batch scoring* on a schedule or when new data lands, writing to a
Delta table. A real-time Model Serving endpoint is possible but serves no
stated consumer.

**D3. Which model goes to production.** The refined model with the time tag
fits best (R 0.83), but **every production prediction extrapolates the tag**
beyond its training range (D-29). The forecast's seed spread (0.27–0.57, D-28)
is what that looks like.
*Recommend* the Type B, H=10 model **without** the tag as champion, or the
tagged model only with an explicit, monitored limit on extrapolation distance.

**D4. Where new data comes from.** The dataset is static, 1998–2017.
Production implies a feed. Weather is available from Environment Canada
(station 1072692, the source already used). **Acidity comes from Teck's lab
system, and whether that can be reached is unknown.** Without a live acidity
feed there is no ground truth, so performance cannot be monitored, only input
and prediction drift.

**D5. CI/CD host and authentication.** The repository is on GitHub
(`Huangc495/EquitySilver`), which makes GitHub Actions the natural choice;
Azure DevOps is the alternative.
*Recommend* GitHub Actions authenticating to Databricks as a **service
principal**, preferably through OIDC workload identity federation rather than
a long-lived secret.

**D6. Should the repository stay public?** It is public today. Two things
argue for reconsidering before production configuration is added:

- `reports/` holds Teck drainage chemistry: the full acidity series, quartiles
  and per-year counts. The published paper's Fig. 4 covers much of it, but
  ours is more granular. **This was flagged to the user and remains
  unresolved.**
- An MLOps build adds workspace URLs, service principal IDs and job
  definitions. None is a secret, but together they map the deployment.

---

## 4. A proposed architecture

A starting point for discussion once D1–D6 are settled, assuming Databricks
is chosen. **Databricks renames and reshapes these products often; check the
current names and APIs against its documentation before building.** This
section may be out of date.

| MLOps practice | Proposed component | Notes specific to this project |
|---|---|---|
| Experiment tracking | MLflow in the workspace | Already wired: every run logs config, seed, split sizes and per-split metrics. Add the git commit SHA as a tag. |
| Model packaging | MLflow **pyfunc** wrapping the LSTM **and its scaler** | The scaler is part of inference. Shipping weights without it causes training/serving skew. Input: raw daily weather; output: mg/L. |
| Model registry | MLflow registry in Unity Catalog, e.g. `equity_silver_databricks_mlops.<schema>.acidity_bd` | Champion and challenger aliases. One model per station. |
| Training pipeline | Databricks Job on an ML job cluster | Tasks: preflight → ingest and validate → train → evaluate → register |
| Promotion gate | Challenger against champion | Compare **R and RMSE in mg/L**, never normalised MSE (D-18); on **held-out years** (blocked split), not all samples (D-22); across **several seeds**, not the best one (D-28). |
| CI | GitHub Actions | Lint and unit tests on every pull request, on synthetic data; integration tests on Databricks. |
| CD | Databricks Asset Bundles (`databricks.yml`) | dev, staging and prod targets. `databricks bundle init mlops-stacks` scaffolds this layout. |
| Batch inference | Scheduled job scoring with the champion | Writes predictions to a Delta table in Unity Catalog. |
| Monitoring | Lakehouse Monitoring on the prediction and feature tables | Precipitation and temperature drift; prediction drift; **window completeness rate**; performance once lab results arrive. |
| Data versioning | Delta tables for raw and processed data | Replaces parquet in the volume; time travel gives reproducible training sets. |

### Monitoring notes that are specific to this model

- **Labels arrive late.** Acidity is measured roughly twice a month, so
  performance monitoring lags. Input and prediction drift are the early
  signals.
- **Missing weather is the main data-quality risk.** 13.7% of historical days
  have a gap, and a sample is dropped if any day in its 70-day window is
  missing. Monitor the share of samples that survive the window rule.
- **Every metric is noisy.** About 185 samples per station, a 30-sample test
  fold, and seed-to-seed SD of 0.03 MSE. Set drift and performance thresholds
  conservatively, or alerts will fire on noise.
- **The time tag, if shipped, must be monitored as a feature.** Its
  normalised value in production is already beyond anything seen in training
  (+2.55 to +2.97 against a training maximum of +2.53).

---

## 5. Concrete code gaps

Each was checked in the source.

| Gap | Where | What is needed |
|---|---|---|
| No model artifact logged | `experiments._log_scenario`, `forecast._log_forecast` | `mlflow.pyfunc.log_model` with a wrapper holding weights, scaler and config; a signature and input example |
| No packaging | repo root | `pyproject.toml`, so jobs install a wheel instead of editing `sys.path` in 8 notebooks |
| Tests need the real data | 71 tests in 9 files (below) | pytest markers separating unit from integration; CI runs unit tests only |
| Reports written into the repository | `cfg.reports_dir` | In production, outputs go to Unity Catalog volumes or tables, not git |
| Processed data is parquet in a volume | `preprocess.build_processed` | Delta tables for lineage and time travel |
| `requirements.txt` unconfirmed | D-01 | Run `00a_cluster_preflight` on the ML cluster; it prints the installed versions as a requirements block |
| `resplit_each_repeat` has no caller | `train.train_repeats`, D-15 | Minor; it raises rather than silently misbehaving |

**Tests that break without data**, by file (errors plus failures):

| File | Count |
|---|---|
| `test_forecast.py` | 23 |
| `test_experiments.py` | 14 |
| `test_phases78.py` | 12 |
| `test_preprocess.py` | 9 |
| `test_windows.py` | 4 |
| `test_splits.py` | 4 |
| `test_preflight.py` | 3 |
| `test_train.py` | 1 |
| `test_config.py` | 1 |

The pure unit tests (`test_models.py`, `test_scaling.py`, `test_report_phase11.py`
and most of `test_windows.py` and `test_train.py`) already use synthetic data
and are CI-ready. The data-dependent tests are valuable: they lock the
paper's sample counts. Keep them, mark them, and run them where the data
lives.

---

## 6. Traps already hit once

Each cost time in the first session. Several will reappear in MLOps work.

**Modelling and evaluation**

- **Normalised MSE only compares within one scaler (D-18).** Refitting the
  scaler on training rows appeared to *improve* MSE by 0.052; R moved +0.002
  and RMSE by 3 mg/L. The models were identical and only the units changed.
  **Any promotion gate or monitor that compares normalised MSE across
  retrainings will be wrong.** Use R and RMSE in mg/L.
- **All-sample metrics and best-of-5 reward overfitting (D-22).** About 68% of
  "all samples" are training rows, and selecting on them picks the most
  overfit run. This inflated the C7 baseline's R from an honest 0.39 to 0.78.
- **A single best seed misleads (D-28).** Forecast MSE across five seeds ran
  0.265 to 0.569. The headline was the luckiest run.
- **The time tag extrapolates in any forward use (D-29).** See D3 above.
- **Type B keeps only about half the samples (Q-01)**, because a 70-day window
  must be complete. Changing the input window changes the sample set.

**Databricks**

- **Serverless has no PyTorch or MLflow**, no runtime selector and no cluster
  environment variables. Use an ML runtime. ML builds report
  `…-cpu-ml-scala2.12`; the preflight checks for this and recognises
  serverless.
- **`information_schema` is read-only system metadata** in every Unity
  Catalog catalog and cannot hold volumes. The user first supplied it as the
  schema.
- **Volumes, Workspace and DBFS are three separate storage systems.** Files in
  one are invisible from the others, and volumes do not appear in the
  Workspace browser. The legacy "Upload data" UI writes to DBFS.
- **Volume paths are `/Volumes/<catalog>/<schema>/<volume>/<path>`.**
  `raw` and `processed` are volumes, not folders. Upload files into the
  volume root, not a nested folder.
- **The user is working through the Databricks UI for the first time.** Give
  exact click paths, and prefer JSON specs they can paste over forms to fill
  in. When their screen might differ from what you expect, ask what they see.

**Repository and environment**

- **The repository is public.** Never commit personal identifiers, the paper
  PDF, or data. The workspace username comes from `${DATABRICKS_USERNAME}`,
  and a test fails if an email address appears in the tracked config. Git
  history was rewritten twice to remove the PDF and an email address, both
  **before** the first push. **Never rewrite or force-push history now.**
- **`/Volumes/...` was once re-rooted to `C:\Volumes\...` on Windows (D-30)**,
  because `Path.is_absolute()` is false there for a bare leading slash. Fixed
  in `config._resolve_against`. Watch for the same bug in new path handling.
- **`python` on PATH is an unrelated agent virtualenv without pip.** Always use
  `.venv/Scripts/python.exe` (Python 3.12.13, created with `uv`) or `uv`
  itself.
- **Large Bash heredocs containing Python have failed** with
  `unexpected EOF while looking for matching '`. Use the Write tool for new
  source files.
- **Run pytest as** `.venv/Scripts/python.exe -m pytest tests/ -q -p no:cacheprovider`.
  Without the flag, pytest warns that it cannot write its cache directory.
- **CRLF warnings on commit are harmless.** `.gitattributes` normalises line
  endings to LF, because development is on Windows and the cluster is Linux.

---

## 7. The development machine

| Tool | State |
|---|---|
| OS and shells | Windows 11; Git Bash and PowerShell |
| Python | `.venv/` on 3.12.13, packages pinned in `requirements.txt` |
| `uv` | installed at `~/.local/bin` |
| Azure CLI | installed, version 2.90.0; login state not checked |
| Databricks CLI | **not installed** |
| GitHub CLI (`gh`) | **not installed** |
| Git | remote `origin` is `https://github.com/Huangc495/EquitySilver.git`; pushes authenticate through Windows Credential Manager; the author identity is set repo-locally |
| Scratch files | use the session scratchpad directory, never the repo |

Installing the Databricks CLI would let you check volumes, jobs and runs
directly instead of relaying through the user. **Ask before installing it or
configuring credentials.**

---

## 8. Repository map

```
CLAUDE.md                replication spec; working rules still apply
HANDOVER.md              this file
DEVIATIONS.md            D-00 to D-31, Q-01 to Q-03; continue at D-32
requirements.txt         pinned to DBR 16.4 LTS ML (unconfirmed, D-01)
configs/base.yaml        every setting; local and Databricks path sets
docs/README.md           Databricks setup and run order
docs/architecture.html   interactive module dependency map
src/acidity_lstm/
  config.py              YAML, local vs Databricks paths, ${VAR} expansion
  io.py                  reads the 21 weather CSVs and the acidity workbook
  preprocess.py          daily reindex, flags, duplicates; never imputes
  windows.py             Type A and B windows, time tag, SampleSet
  splits.py              random and blocked (whole-year) splits
  scaling.py             z-score with population std
  models.py              LSTMRegressor, FCBaseline, MATLAB-style init
  train.py               SGDM loop, early stopping, repeats, best-of-5
  evaluate.py            MSE, R, RMSE in mg/L, figures
  experiments.py         grids, variants, MLflow logging   <- the hub
  forecast.py            phases 9-10: forecast and sensitivity
  preflight.py           cluster, data and package checks
  audit.py               phase 0-2 reports; shared markdown tables
  report_phase*.py       one report module per phase
notebooks/
  00a_cluster_preflight  run first; changes nothing
  00_data_audit          writes the processed parquet the rest read
  01 to 06               phases 6 to 11
tests/                   264 tests; 71 need the real data
reports/                 generated markdown and figures (contains Teck data)
data/                    gitignored; raw/ holds the 22 input files
```

`experiments.py` is the hub: it imports 6 modules and 10 import it. For the
full call graph, open `docs/architecture.html`.

---

## 9. Suggested plan

A proposal only. The user works phase by phase and approves each one.

| Phase | Goal | Done when |
|---|---|---|
| **M0** | Databricks baseline | The classic ML cluster exists, the preflight is fully green, notebooks 00–06 run on Databricks and reproduce the local numbers, and D-01 is closed from the cluster's actual versions |
| **M1** | Packaging and CI | `pyproject.toml`; unit and integration tests separated by markers; GitHub Actions running lint and unit tests on every pull request |
| **M2** | A registrable model | A pyfunc bundling weights, scaler and config, with a signature, registered in Unity Catalog |
| **M3** | Training pipeline and CD | An Asset Bundle deploying a training job to dev, staging and prod, with the promotion gate from section 4 |
| **M4** | Batch inference | A scheduled job scoring with the champion into a Delta table |
| **M5** | Monitoring | Drift, window completeness and delayed-label performance, with alert thresholds |
| **M6** | Data ingestion | Depends on D4: an automated weather feed at least, and acidity if the lab system is reachable |
| **M7** | Runbook | How to retrain, promote, roll back and respond to an alert |

M0 comes first because every later phase assumes the pipeline runs on
Databricks, and that has not yet been shown.

---

## 10. Working agreements with this user

These held throughout the first session.

- **Phase by phase.** At each stop, summarise what was done, the key numbers
  and the open questions, then wait for approval.
- **Every setting lives in config**, and every departure from the plan is
  recorded in `DEVIATIONS.md` with its reason and likely effect. The
  replication report parses that file, and `report_phase11.LIKELY_EFFECT`
  must gain an entry for each new deviation, or the report generator raises.
- **No tuning to hit the paper's numbers.** Report non-replications honestly;
  D-21 is the example.
- **Check the claim before reporting it.** Several "improvements" in the first
  session were artefacts: D-18, D-22 and the C7 baseline.
- **Commit with detailed messages and push to GitHub.** The user has accepted
  this workflow. Pushing publishes to a public repository, so run the
  pre-flight scan for data, PDFs and personal identifiers first.
- **Raise the concern once, then do what the user decides.** They chose to
  remove the PDF, to scrub the email address, and to use the `default`
  schema. The one open concern is D6, the Teck data in `reports/`.
