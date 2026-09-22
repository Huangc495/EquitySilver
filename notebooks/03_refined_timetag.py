# Databricks notebook source
# MAGIC %md
# MAGIC # Phase 8 - Refined model with the time tag
# MAGIC
# MAGIC Adds the day number as a third input feature (BD, Type B, H = 10) and
# MAGIC compares it with the original model. Runs both time-tag variants
# MAGIC (`per_step` and `constant`), plus a blocked-split check of whether the
# MAGIC tag's gain survives extrapolation to unseen years.
# MAGIC
# MAGIC Writes `reports/refined_timetag.md` and Figs 9-10.

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
from acidity_lstm.experiments import run_refined, scenarios_to_table, setup_mlflow
from acidity_lstm.preprocess import clean_acidity, clean_weather
from acidity_lstm.report_phase8 import make_figures, write_report

print("Running on Databricks:", on_databricks())
cfg = load_config(REPO_ROOT / "configs" / "base.yaml")

# COMMAND ----------

weather, _ = clean_weather(cfg)
acidity, _ = clean_acidity(cfg)
mlflow = setup_mlflow(cfg)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paper settings (random split)
# MAGIC
# MAGIC BD is the paper's scenario; C7 is run as a supplementary check.

# COMMAND ----------

scenarios = run_refined(cfg, acidity, weather, mlflow=mlflow, variant="paper")
table = scenarios_to_table(scenarios)
table[["station", "time_tag", "best_mse", "r_all", "mse_test", "r_test"]]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Blocked split
# MAGIC
# MAGIC Holds out whole years, so the time tag must extrapolate to dates the
# MAGIC model never saw - the same demand Phase 9's forecast will make.

# COMMAND ----------

blocked = run_refined(cfg, acidity, weather, mlflow=mlflow, variant="blocked",
                      stations=("BD",))
blocked_table = scenarios_to_table(blocked)
blocked_table[["station", "time_tag", "best_mse", "r_all", "mse_test", "r_test"]]

# COMMAND ----------

# MAGIC %md ## Figures and report

# COMMAND ----------

figs = make_figures(cfg, scenarios)
report = write_report(cfg, scenarios, figs, blocked=blocked)
print(report)

# COMMAND ----------

import pandas as pd

all_rows = pd.concat([table, blocked_table], ignore_index=True)
all_rows.to_csv(cfg.reports_dir / "refined_timetag_raw.csv", index=False)

if mlflow is not None:
    with mlflow.start_run(run_name="phase8-summary"):
        for path in figs.values():
            mlflow.log_artifact(str(path), artifact_path="figures")
        mlflow.log_artifact(str(cfg.reports_dir / "refined_timetag.md"))
        mlflow.log_text(all_rows.to_csv(index=False), "refined_timetag_raw.csv")

# COMMAND ----------

try:
    displayHTML(f"<pre>{report}</pre>")  # noqa: F821  (Databricks builtin)
except NameError:
    pass
