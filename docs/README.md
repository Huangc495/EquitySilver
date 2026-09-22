# Source paper

This repository replicates:

> Ma, L., Huang, C., Liu, Z.-S., Morin, K.A., Aziz, M., Meints, C. (2021).
> *The correlation between drainage chemistry and weather for full-scale waste
> rock piles based on artificial neural network.*
> **Journal of Contaminant Hydrology** 239, 103793.
> https://doi.org/10.1016/j.jconhyd.2021.103793

## The PDF is not tracked

`CLAUDE.md` expects the paper at `docs/Ma_2021_JCH.pdf`, and the code and
reports refer to it by that path. The file is **deliberately not committed**:
it is a copyrighted Elsevier article and this repository is public.

`docs/*.pdf` is gitignored. To work with the paper locally, place your own
copy at:

```
docs/Ma_2021_JCH.pdf
```

Access it through the DOI above, via your institution's subscription, or from
ScienceDirect. Nothing in the pipeline reads the PDF at runtime — it is a
reference for interpreting the replication, so the code runs without it.

## What the paper provides

Every target this replication is measured against is reproduced as data in the
code rather than quoted from the paper:

| Constant | Module | Contents |
|---|---|---|
| `PAPER_TABLE1` | `experiments.py` | Table 1 best-of-5 MSE, both stations, Types A and B, H = 5/10/20 |
| `PAPER_FIG6_R` | `experiments.py` | Fig. 6 correlation at Type B, H = 10 |
| `PAPER_FC_R`, `PAPER_LSTM_R` | `experiments.py` | Section 3.4 fully connected baseline comparison |
| `PAPER_ORIGINAL`, `PAPER_REFINED` | `experiments.py` | Section 3.5 time-tag model |
| `PAPER_FORECAST` | `forecast.py` | Section 3.6 two-year forecast |

`tests/test_report_phase11.py` asserts these match the values recorded in
`CLAUDE.md`, so a typo in a target cannot silently flatter the replication.

See `reports/replication_report.md` for the results and `DEVIATIONS.md` for
every departure from the paper.

## Codebase map

`docs/architecture.html` is an interactive dependency map: all 17 library
modules in call-order layers, which one imports which, the real call chain
behind a parametric-study run, and a table of every module's responsibility
and entry points. Open it in a browser.

The edges are extracted from the source with `ast` rather than drawn by hand,
so the map cannot quietly drift from the code. `config` is drawn as a
foundation bar rather than with sixteen edges into it, since every module
imports it.

---

# Running on Databricks

Workspace: **Equity-Silver-Databricks-MLOps** (Azure Databricks).
Paths and the MLflow experiment are selected automatically from the
`DATABRICKS_RUNTIME_VERSION` environment variable, so the same code runs
locally and on the cluster with no edits.

## 1. Cluster

**The runtime must be a Databricks Runtime ML build.** PyTorch and MLflow ship
only with ML runtimes; a standard runtime of the same version does not have
them, and the preflight will fail with `torch not installed` and
`mlflow not installed`.

Create the cluster as: single node, CPU only, and under **Databricks Runtime
version** pick an entry labelled **ML**, e.g. `16.4 LTS ML`. ML builds tag
themselves in `DATABRICKS_RUNTIME_VERSION` — `16.4.x-cpu-ml-scala2.12` rather
than `16.4.x-scala2.12` — which is how `00a_cluster_preflight` detects this.

The model has about 500 parameters, so no GPU and no Spark parallelism are
needed. Install `openpyxl` if the runtime lacks it.

### Serverless will not work

Serverless compute has no runtime selector, no PyTorch and no cluster
environment variables. `00a_cluster_preflight` recognises it and says so.
Create a classic all-purpose cluster on an ML runtime and attach the
notebooks to that instead.

### The workspace username

Config values may reference `${VAR}`, expanded at load time, so no personal
identifier is stored in this public repo. `DATABRICKS_USERNAME` normally
resolves **automatically** from the signed-in workspace user, so there is
usually nothing to set.

Override it only if you need a different value:

- classic cluster: **Compute > Edit > Advanced options > Spark > Environment
  variables**, `DATABRICKS_USERNAME=your.name@example.com`
- any notebook: `os.environ["DATABRICKS_USERNAME"] = "your.name@example.com"`
  before `load_config()`

## 2. Unity Catalog volumes

The data lives in two volumes under catalog `equity_silver_databricks_mlops`,
schema `default`. Volume paths are
`/Volumes/<catalog>/<schema>/<volume>/<path>`, so `raw` and `processed` are
volumes, not folders:

```sql
CREATE VOLUME IF NOT EXISTS equity_silver_databricks_mlops.default.raw;
CREATE VOLUME IF NOT EXISTS equity_silver_databricks_mlops.default.processed;
```

Note that `information_schema` is read-only system metadata, auto-created in
every catalog, and cannot hold volumes.

## 3. Upload the raw data

`data/` is gitignored — the drainage chemistry is not in this repository — so
upload it once, into the `raw` volume:

```
/Volumes/equity_silver_databricks_mlops/default/raw/
    2017 ARD Chemisty_Clean.xlsx
    en_climate_daily_BC_1072692_1997_P1D.csv
    ... through 2017 (21 files)
```

**Catalog** (left nav) > `equity_silver_databricks_mlops` > `default` >
**Volumes** > `raw` > **Upload to this volume**. Or `databricks fs cp`.

> **Volumes are not in the Workspace browser.** Databricks keeps three
> separate storage areas, and files put in one are invisible from the others:
>
> | Area | Where it is | Path |
> |---|---|---|
> | **Unity Catalog volumes** | left nav **Catalog** | `/Volumes/<catalog>/<schema>/<volume>/` |
> | Workspace | left nav **Workspace** | `/Workspace/Users/<you>/` |
> | DBFS (legacy) | hidden by default | `/FileStore/`, `dbfs:/` |
>
> This pipeline reads from a **volume**, so looking in Workspace for the
> uploaded data will always come up empty. The legacy "Upload data" UI writes
> to DBFS, not to a volume. `00a_cluster_preflight` searches all three and
> tells you which one your files are actually in.

## 4. Clone the repository into Databricks

The notebooks are `.py` files beginning with `# Databricks notebook source`,
so Databricks renders them as notebooks once the repo is cloned. They are not
uploaded by hand.

1. Left nav **Workspace** > your home folder (**Users > you@example.com**).
2. **Create** (top right) > **Git folder**.
3. Git repository URL: `https://github.com/Huangc495/EquitySilver`
   Git provider: **GitHub**. The repository is public, so no token is needed
   to clone and run; one is only required to push from Databricks.
4. **Create Git folder**.

You now have `/Workspace/Users/<you>/EquitySilver/` containing `notebooks/`.
Use **Pull** in that folder to pick up later changes.

## 5. Run

Open a notebook, attach it to the cluster with the dropdown at the top right,
then **Run all**. Each notebook adds `src/` to `sys.path` itself.

| Notebook | Phase | Writes |
|---|---|---|
| `00a_cluster_preflight` | - | nothing; checks the environment and data |
| `00_data_audit` | 0-2 | `reports/00_data_audit.md`, `02_sample_counts.md`, parquet |
| `01_parametric_study` | 6 | `reports/table1.md`, Figs 6-7 |
| `02_fc_baseline` | 7 | `reports/fc_baseline.md` |
| `03_refined_timetag` | 8 | `reports/refined_timetag.md`, Figs 9-10 |
| `04_forecast` | 9-10 | `reports/forecast_sensitivity.md`, Figs 11-12 |
| `05_sensitivity` | 10 | Fig. 12 alone |
| `06_replication_report` | 11 | `reports/replication_report.md` |

Runs log to `/Users/${DATABRICKS_USERNAME}/equity-silver-lstm` in MLflow.

Run **`00a_cluster_preflight` first**. It changes nothing and verifies the
environment variable, the uploaded data and the installed packages in one
pass, failing with a clear message rather than letting a notebook break
part-way through. `00_data_audit` must then run before the rest: it writes
the processed parquet they read.

## 6. Check it first

```bash
pytest
```

CLAUDE.md rule 6: the suite must pass before any experiment phase runs. It
needs the raw data in place, since several tests assert against the real
sample counts.
