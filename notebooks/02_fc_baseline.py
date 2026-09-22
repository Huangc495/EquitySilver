# Databricks notebook source
# MAGIC %md
# MAGIC # Phase 7 - Fully connected baseline
# MAGIC
# MAGIC Trains the FC baseline of Ma et al. (2020a) on the Type B input, using
# MAGIC the same splits, training settings and repeats as the LSTM, and compares
# MAGIC R for BD and C7.
# MAGIC
# MAGIC Writes `reports/fc_baseline.md`.

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
from acidity_lstm.experiments import (
    run_fc_baseline,
    scenarios_to_table,
    setup_mlflow,
)
from acidity_lstm.preprocess import clean_acidity, clean_weather
from acidity_lstm.report_phase7 import make_figure, write_report

print("Running on Databricks:", on_databricks())
cfg = load_config(REPO_ROOT / "configs" / "base.yaml")

# COMMAND ----------

weather, _ = clean_weather(cfg)
acidity, _ = clean_acidity(cfg)
mlflow = setup_mlflow(cfg)

# COMMAND ----------

# MAGIC %md ## Train both models on the Type B input

# COMMAND ----------

scenarios = run_fc_baseline(cfg, acidity, weather, mlflow=mlflow)
table = scenarios_to_table(scenarios)
table[["station", "kind", "best_mse", "r_all", "r_test", "rmse_mgL"]]

# COMMAND ----------

# MAGIC %md ## Figure and report

# COMMAND ----------

fig_path = make_figure(cfg, scenarios)
report = write_report(cfg, scenarios, fig_path)
print(report)

# COMMAND ----------

table.to_csv(cfg.reports_dir / "fc_baseline_raw.csv", index=False)

if mlflow is not None:
    with mlflow.start_run(run_name="phase7-summary"):
        mlflow.log_artifact(str(fig_path), artifact_path="figures")
        mlflow.log_artifact(str(cfg.reports_dir / "fc_baseline.md"))
        mlflow.log_text(table.to_csv(index=False), "fc_baseline_raw.csv")

# COMMAND ----------

try:
    displayHTML(f"<pre>{report}</pre>")  # noqa: F821  (Databricks builtin)
except NameError:
    pass
