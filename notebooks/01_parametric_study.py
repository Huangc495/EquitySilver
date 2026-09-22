# Databricks notebook source
# MAGIC %md
# MAGIC # Phase 6 - Parametric study
# MAGIC
# MAGIC Reproduces paper Table 1 and Figs 6-7, plus three robustness variants:
# MAGIC blocked splits, a common sample set, and a training-only scaler.
# MAGIC
# MAGIC Writes `reports/table1.md` and figures under `reports/figures/`, and
# MAGIC logs every run to MLflow.

# COMMAND ----------

import logging
import sys
from pathlib import Path

REPO_ROOT = Path.cwd()
while not (REPO_ROOT / "configs" / "base.yaml").exists() and REPO_ROOT != REPO_ROOT.parent:
    REPO_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logging.getLogger("acidity_lstm.splits").setLevel(logging.WARNING)

from acidity_lstm.config import load_config, on_databricks
from acidity_lstm.experiments import log_study_artifacts, run_parametric_study
from acidity_lstm.preprocess import clean_acidity, clean_weather
from acidity_lstm.report_phase6 import make_figures, write_table1

print("Running on Databricks:", on_databricks())
cfg = load_config(REPO_ROOT / "configs" / "base.yaml")

# COMMAND ----------

# MAGIC %md ## Load the cleaned data

# COMMAND ----------

weather, _ = clean_weather(cfg)
acidity, _ = clean_acidity(cfg)
print("weather:", weather.shape, "| acidity:", acidity.shape)
print(acidity.groupby("station").size().to_dict())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run the grid
# MAGIC
# MAGIC 4 variants x 12 scenarios x 5 repeats = 240 runs. Sequential on CPU.

# COMMAND ----------

table, scenarios = run_parametric_study(
    cfg,
    variants=("paper", "blocked", "common", "fit_train"),
    acidity=acidity,
    weather=weather,
)
print(table.shape)
table.head()

# COMMAND ----------

# MAGIC %md ## Figures and report

# COMMAND ----------

figs = make_figures(cfg, scenarios)
report = write_table1(cfg, table, scenarios, figs)
print(report)

# COMMAND ----------

# The tidy grid is saved alongside the report for later phases.
out_csv = cfg.reports_dir / "table1_raw.csv"
table.to_csv(out_csv, index=False)
print("wrote", out_csv)

# Figures and reports as MLflow artifacts (CLAUDE.md rule 7).
log_study_artifacts(cfg, table, figs, report_path=cfg.reports_dir / "table1.md")

# COMMAND ----------

try:
    displayHTML(f"<pre>{report}</pre>")  # noqa: F821  (Databricks builtin)
except NameError:
    pass
