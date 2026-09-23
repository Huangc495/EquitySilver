# MLOps stage: decisions, plan and status

The replication of Ma et al. (2021) is complete (`CLAUDE.md` phases 0–11,
`reports/replication_report.md`). This stage runs the same study on Azure as a
production ML system. `HANDOVER.md` describes the starting state; this file
records what the user decided and where the work stands. Where the two
disagree, this file is newer.

`CLAUDE.md`'s working rules still apply: phase by phase with a stop at each
phase, every setting in config, every seed logged, notebooks thin, the same
code locally and on Databricks.

Decisions here are numbered **D1–D7**; D1–D6 follow `HANDOVER.md` section 3.
They are separate from the deviations `D-00` onwards in `DEVIATIONS.md`.

---

## Decisions (settled 2026-09-22)

D1–D6 were settled at the start of the stage. D7 was added during M0, when
the workspace turned out to have no classic compute.

| # | Question | Decision | Handover recommendation |
|---|---|---|---|
| D1 | Platform | **Databricks-native**: MLflow, Unity Catalog registry, Jobs, Asset Bundles, Lakehouse Monitoring | same |
| D2 | What "production" means | **Scheduled batch scoring** into a Delta table | same |
| D3 | Which model ships | **Untagged champion, tagged challenger in shadow** | untagged champion |
| D4 | Where new data comes from | **Live weather, acidity by upload** | same |
| D5 | CI/CD host | **Azure DevOps Pipelines** | GitHub Actions |
| D6 | Repository visibility | **Make it private** | same |
| D7 | Compute | **Serverless with a pinned environment** | *(not foreseen: the handover assumed a classic ML cluster)* |

### D1 Databricks-native

Everything stays in workspace **Equity-Silver-Databricks-MLOps**, which already
holds the data in Unity Catalog (`equity_silver_databricks_mlops.default`). No
Azure Machine Learning workspace.

### D2 Scheduled batch scoring

A job scores with the champion on a schedule, or when new data lands, and
writes predictions to a Delta table in Unity Catalog. There is no real-time
endpoint, because acidity is sampled about twice a month and no consumer needs
a sub-second answer.

### D3 Untagged champion, tagged challenger in shadow

- **Champion:** Type B, H=10, **without** the time tag, one model per station
  (BD, C7). No production input extrapolates.
- **Shadow challenger:** the refined model **with** the time tag. It is scored
  alongside the champion on every batch and written to the same predictions
  table, but never served as the answer.
- **Why shadow:** the tagged model fits better (R 0.83 against 0.67 at BD),
  but every production prediction extrapolates the day-number tag beyond its
  training range (`DEVIATIONS.md` D-29). Running it in shadow measures what
  that extrapolation costs, on real lab results as they arrive, without
  exposing anyone to it.
- **Consequence for monitoring:** the tagged model's normalised time tag and
  its distance beyond the training maximum are logged as a monitored feature.
  Promoting the challenger needs evidence from the shadow record, not its
  training fit.

### D4 Live weather, acidity by upload

- **Weather:** an automated daily pull from Environment Canada, climate
  station 1072692, which is the source the replication already used.
- **Acidity:** arrives as files dropped into the `raw` volume, uploaded by
  hand or exported from the lab. Each upload supplies the delayed labels for
  performance monitoring.
- **Consequence:** until acidity files arrive, only input drift and
  prediction drift can be monitored. Performance monitoring lags sampling by
  weeks.

### D5 Azure DevOps Pipelines

The user chose Azure DevOps over the recommended GitHub Actions.

- **Authentication:** an Azure Resource Manager **service connection using
  workload identity federation**, backed by a Microsoft Entra service
  principal that is added to the Databricks workspace. Pipelines get a
  short-lived Entra token through an `AzureCLI@2` task, and the Databricks CLI
  uses it with `DATABRICKS_AUTH_TYPE=azure-cli`. No long-lived secret is
  stored. This is the Azure DevOps equivalent of GitHub's OIDC.
- **Code stays on GitHub** (decided at M1). Azure Pipelines reads it
  through the Azure Pipelines GitHub App. No Azure DevOps organization
  existed, so the user creates one (see "M1: Azure DevOps setup").

### D6 Make the repository private

The user makes `github.com/Huangc495/EquitySilver` private: **Settings >
General > Danger Zone > Change repository visibility**. Only the owner can do
this. It closes the open concern about Teck drainage chemistry in `reports/`.

Consequences:

- **Anything already scraped stays public.** Git history is not rewritten.
- **The Databricks Git folder needs credentials to pull.** Link GitHub under
  **User Settings > Linked accounts** in the workspace, or pulls will fail
  once the repo is private.
- **Azure Pipelines needs access too**, through its GitHub service connection
  (see D5).
- **GitHub Free cannot enforce required checks on a private repository.**
  Branch protection and rulesets on private repositories need GitHub Pro.
  CI still runs and reports on every pull request; it just cannot block a
  merge.
- **Until the repository is confirmed private, commit no deployment
  identifiers:** no workspace URL, service principal ID or Azure DevOps
  organization. Keep them in environment variables or pipeline variables, as
  with `${DATABRICKS_USERNAME}`.

### D7 Serverless with a pinned environment

The workspace has **no classic compute plane**. Every cluster request fails
with "does not have any associated worker environments", so the handover's
Runtime ML cluster, all-purpose or job, cannot exist here. The user chose
serverless over creating a new workspace with classic compute.

- **Environment:** serverless environment 4 (Python 3.12.3), plus
  `requirements-databricks.txt` for what it lacks: a CPU-only torch 2.6.0
  wheel, mlflow 2.21.3 and openpyxl. A probe run confirmed that all three
  install, that the data and volumes resolve, and that the MLflow experiment
  resolves without setup.
- **Local pins follow it:** numpy 2.1.3 and the rest of environment 4
  (`DEVIATIONS.md` D-33, closing D-01).
- **Consequences for later phases:** jobs are serverless tasks with an
  `environments` block, never a `job_clusters` block. That covers M0 now, and
  the Asset Bundle in M3 and the scoring job in M4 later. There are no
  clusters to size, start or terminate.

---

## Plan

The phases from `HANDOVER.md` section 9, adjusted for the decisions above.

| Phase | Goal | Done when | Status |
|---|---|---|---|
| **M0** | Databricks baseline | Preflight fully green on serverless; notebooks 00–06 run on Databricks and reproduce the local numbers; D-01 closed from the platform's actual versions | **done** (approved 2026-09-22) |
| **M1** | Packaging and CI | `pyproject.toml`; unit and integration tests separated by pytest markers; an **Azure Pipeline** running lint and unit tests on every pull request | **built and verified locally; waiting on the Azure DevOps organization** |
| M2 | A registrable model | A pyfunc bundling weights, scaler and config, with a signature, registered in Unity Catalog; champion and shadow challenger per D3 | not started |
| M3 | Training pipeline and CD | An Asset Bundle deploying a training job to dev, staging and prod **from Azure Pipelines**, with the promotion gate below | not started |
| M4 | Batch inference | A scheduled job scoring champion **and** shadow challenger into one Delta table | not started |
| M5 | Monitoring | Input drift, prediction drift, window completeness, time-tag extrapolation distance, and delayed-label performance, with alert thresholds | not started |
| M6 | Data ingestion | Daily Environment Canada pull for station 1072692; acidity uploads to the `raw` volume picked up and validated | not started |
| M7 | Runbook | How to retrain, promote, roll back and respond to an alert | not started |

### The promotion gate (unchanged from the handover)

A challenger replaces the champion only if it wins on **R and RMSE in mg/L**,
never normalised MSE (D-18); on **held-out years** from the blocked split, not
all samples (D-22); and across **several seeds**, not the best one (D-28).

---

## M0 checklist

| Step | Who | State |
|---|---|---|
| Install the Databricks CLI | Claude | done: v1.17.0 via winget |
| Sign the CLI in to the workspace (browser OAuth) | user | done: profile `equity-silver` |
| Pull the Git folder to current `main` | Claude | done (it was 4 commits behind) |
| Create a classic ML cluster | Claude | **impossible**: no classic compute plane (D-33, D7) |
| Probe serverless with the preflight | Claude | done: everything passed except the old Runtime ML rule |
| Close D-01: re-pin `requirements.txt` from the platform | Claude | done: serverless environment 4 |
| Run notebooks 00–06 on serverless | Claude | **done**: all 8 tasks succeeded |
| Compare numpy 2 locally with the committed numpy 1 reports | Claude | **done**: bit-identical |
| Compare Databricks with local | Claude | **done**: identical to floating-point noise |
| Confirm MLflow runs appear in `/Users/<you>/equity-silver-lstm` | Claude | **done**: 378 runs |

## M0 results (2026-09-22)

The first end-to-end run on Databricks was a one-time serverless job,
`equity-silver-m0`. It ran eight notebook tasks in sequence, sharing one
environment built from `requirements-databricks.txt`, from the Git folder
at commit `f56eadb`. **All eight succeeded**, in about 11 minutes of compute:

| Task | Seconds |
|---|---|
| 00a preflight (including installing the environment) | 69 |
| 00 data audit | 29 |
| 01 parametric study | 277 |
| 02 FC baseline | 51 |
| 03 time tag | 76 |
| 04 forecast | 41 |
| 05 sensitivity | 30 |
| 06 replication report | 91 |

**The numbers reproduce, and they were checked two ways.**

| Comparison | Result |
|---|---|
| Local on numpy 2.1.3 against the committed numpy 1.26.4 reports | all 66 result rows **bit-identical**; only wall-clock seconds and figure rendering differ |
| Databricks (Linux, serverless) against local (Windows) | chosen seeds, epochs and sample counts identical; largest MSE difference 1.3e-7, largest RMSE difference 0.00016 mg/L, sensitivity means within 0.001 mg/L |
| Rounded reports: Table 1, FC baseline, time tag, forecast and sensitivity, sample counts | **identical, line for line** |

One difference was real. The data audit's "longest gap runs" table
ordered tied lengths three different ways: numpy 1 on Windows, numpy 2 on
Windows, and numpy 2 on Linux. Ties are now broken by start date, so every
platform writes the same file.

**MLflow:** the workspace experiment holds 378 runs, all `FINISHED`, all
tagged with serverless `client.4.10`. The 310 training runs each log their
seed and split sizes. The summary runs carry the figures and reports.

**Reports** from the cluster are in
`/Volumes/equity_silver_databricks_mlops/default/processed/reports` (D-32).
They are not committed, and must not be: the cluster's copy of the
replication report names the MLflow experiment by its full path, which
includes the workspace user's email.

### Tooling notes for later phases

- **The CLI hides some errors behind retries.** `clusters create` simply
  hung. `--debug` with a timeout showed the real error.
- **Git Bash rewrites leading-slash arguments.** `/Users/...` became
  `C:/Program Files/Git/Users/...`. Set `MSYS_NO_PATHCONV=1`, or use
  PowerShell, for any workspace path.
- **`experiments search-runs` fails** on a bug in the Databricks Go SDK. The
  raw `databricks api post /api/2.0/mlflow/runs/search` works.
- **The `equity-silver` profile's host includes a browser path**
  (`…/browse/folders/…`). The Go CLI tolerates it. The Python SDK, and
  therefore MLflow run from this machine, does not. It needs fixing before
  M2 and M3.

---

## M1 results (2026-09-22)

| Piece | What it does |
|---|---|
| `pyproject.toml` | Makes `acidity-lstm` an installable package: `src/` layout, version from `acidity_lstm.__version__`, dependency floors. Exact pins stay in `requirements.txt`. Also holds the pytest and ruff settings. |
| `requirements-dev.txt` | `requirements.txt` plus `ruff==0.16.8` |
| `integration` marker | 71 tests that read the real data; CI deselects them (`DEVIATIONS.md` D-34) |
| `tests/conftest.py` | Skips integration tests with a reason when the data is missing; `ACIDITY_REQUIRE_DATA=1` makes that a failure instead |
| ruff | An explicit rule set (`F`, `E4`, `E7`, `E9`), so an upgrade cannot widen it silently. 14 unused imports and empty f-strings were fixed. |
| `azure-pipelines.yml` | On each pull request into `main` and each push to it: CPU PyTorch, the pinned dependencies, lint, unit tests with published results, a wheel as a pipeline artifact |

**Verified** by cloning the repo fresh, so no `data/` is present as on a CI
agent, and running the pipeline's exact commands in a new Python 3.12 venv.
Every step passed in about 2.5 minutes: 200 unit tests passed and 71 were
deselected. torch stayed on the CPU build throughout, and the wheel built.
The full local suite, data included, passes all 271 tests.

**A packaging bug found and fixed.** An installed (non-editable) wheel
lives in site-packages, where `load_config()` looked for
`configs/base.yaml` and failed. Worse, `repo_root` pointed into
site-packages even when a config path was passed. That would have broken
the preflight's `requirements.txt` check and the report's `DEVIATIONS.md`
parsing as soon as a job installed the wheel. `repo_root` now comes from
the config's own location, `<root>/configs/base.yaml`. Every notebook
already passes that path explicitly, so they work with an installed wheel
unchanged. A test covers it.

**Not yet shown:** the pipeline running in Azure DevOps. That needs the
organization below.

## M1: Azure DevOps setup (user)

1. **Create the organization.** Go to <https://dev.azure.com> and sign in
   with the Microsoft account you use for Azure. Choose **New
   organization**, pick a name, set the region to **Canada Central**, and
   continue.
2. **Create a project** named `EquitySilver`, visibility **Private**.
3. **Get a build agent.** A new organization has no free
   Microsoft-hosted agent. Choose one of:
   - **Request the free grant** at <https://aka.ms/azpipelines-parallelism-request>.
     It takes about 2–3 business days. After that, `azure-pipelines.yml`
     runs as written.
   - **Run a self-hosted agent on this PC.** It is free and works
     immediately. Go to **Project settings > Agent pools > Default > New
     agent > Windows**. `pool:` then becomes `name: Default`, and the agent
     needs Python 3.12 in its tool cache.
   - **Buy a Microsoft-hosted parallel job**, about US$40/month, under
     **Organization settings > Billing**.
4. **Create the pipeline.** Go to **Pipelines > Create Pipeline > GitHub
   (YAML)**. Authorize the **Azure Pipelines** GitHub App, and grant it
   *only* `Huangc495/EquitySilver`. Select the repository, choose
   **Existing Azure Pipelines YAML file**, set branch `main` and path
   `/azure-pipelines.yml`, then **Run**.
5. **Optional: require the check on GitHub.** Go to **Settings > Rules >
   Rulesets > New branch ruleset** on `main` and require the pipeline's
   status check. This only works on a private repository with GitHub Pro
   (see D6).

