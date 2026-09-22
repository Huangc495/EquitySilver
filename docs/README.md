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

---

# Running on Databricks

Workspace: **Equity-Silver-Databricks-MLOps** (Azure Databricks).
Paths and the MLflow experiment are selected automatically from the
`DATABRICKS_RUNTIME_VERSION` environment variable, so the same code runs
locally and on the cluster with no edits.

## 1. Cluster

Single node, CPU only, Databricks Runtime **ML (LTS)** — PyTorch and MLflow
are preinstalled. The model has about 500 parameters, so no GPU and no Spark
parallelism are needed. Install `openpyxl` if the runtime lacks it.

Set one cluster environment variable
(**Compute > Edit > Advanced options > Spark > Environment variables**):

```
DATABRICKS_USERNAME=your.name@example.com
```

Config values may reference `${VAR}`; they are expanded at load time and raise
a named error if unset, so no personal identifier is stored in this public
repo.

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

Catalog Explorer > the `raw` volume > **Upload to this volume**, or
`databricks fs cp`.

## 4. Run

Clone this repository into a Databricks Git folder and run the notebooks in
order. Each adds `src/` to `sys.path` itself.

| Notebook | Phase | Writes |
|---|---|---|
| `00_data_audit` | 0-2 | `reports/00_data_audit.md`, `02_sample_counts.md`, parquet |
| `01_parametric_study` | 6 | `reports/table1.md`, Figs 6-7 |
| `02_fc_baseline` | 7 | `reports/fc_baseline.md` |
| `03_refined_timetag` | 8 | `reports/refined_timetag.md`, Figs 9-10 |
| `04_forecast` | 9-10 | `reports/forecast_sensitivity.md`, Figs 11-12 |
| `05_sensitivity` | 10 | Fig. 12 alone |
| `06_replication_report` | 11 | `reports/replication_report.md` |

Runs log to `/Users/${DATABRICKS_USERNAME}/equity-silver-lstm` in MLflow.
`00_data_audit` must run first: it writes the processed parquet the rest read.

## 5. Check it first

```bash
pytest
```

CLAUDE.md rule 6: the suite must pass before any experiment phase runs. It
needs the raw data in place, since several tests assert against the real
sample counts.
