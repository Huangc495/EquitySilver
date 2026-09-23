# Databricks notebook source
# MAGIC %md
# MAGIC # Phase 0 - Data audit
# MAGIC
# MAGIC Audits the Equity Silver weather CSVs and acidity workbook before any
# MAGIC modelling. Writes `reports/00_data_audit.md` and figures under
# MAGIC `reports/figures/`.
# MAGIC
# MAGIC All logic lives in `src/acidity_lstm/audit.py`; this notebook only
# MAGIC loads config, calls it, and displays the result.

# COMMAND ----------

import sys
from pathlib import Path

# Make `src/` importable both locally and in a Databricks Git folder.
REPO_ROOT = Path.cwd()
while not (REPO_ROOT / "configs" / "base.yaml").exists() and REPO_ROOT != REPO_ROOT.parent:
    REPO_ROOT = REPO_ROOT.parent
try:
    import acidity_lstm  # noqa: F401  installed as a wheel (Asset Bundle job)
except ImportError:
    sys.path.insert(0, str(REPO_ROOT / "src"))  # a Git folder or a plain checkout

import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

from acidity_lstm.audit import run_audit
from acidity_lstm.config import load_config, on_databricks
from acidity_lstm.preprocess import build_processed

print("Running on Databricks:", on_databricks())

# COMMAND ----------

cfg = load_config(REPO_ROOT / "configs" / "base.yaml")
print("acidity workbook:", cfg.resolve("acidity_excel"))
print("weather glob:    ", cfg.resolve("weather_glob"))
print("processed dir:   ", cfg.resolve("processed_dir"))
print("reports dir:     ", cfg.reports_dir)

# COMMAND ----------

# MAGIC %md ## Run the audit

# COMMAND ----------

report = run_audit(cfg)
print(report)

# COMMAND ----------

# MAGIC %md ## Write the cleaned parquet outputs (Phase 1)

# COMMAND ----------

weather, acidity, wrep, arep = build_processed(cfg, save=True)
print("weather:", weather.shape, "| acidity:", acidity.shape)
print(acidity.groupby("station").size())

# COMMAND ----------

try:
    displayHTML(f"<pre>{report}</pre>")  # noqa: F821  (Databricks builtin)
except NameError:
    pass

# COMMAND ----------

# MAGIC %md ## Phase 2 - final sample counts

# COMMAND ----------

from acidity_lstm.audit import run_sample_count_report

counts_report = run_sample_count_report(cfg)
print(counts_report)
