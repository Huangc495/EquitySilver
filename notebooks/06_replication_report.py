# Databricks notebook source
# MAGIC %md
# MAGIC # Phase 11 - Replication report
# MAGIC
# MAGIC Re-runs every experiment and assembles `reports/replication_report.md`:
# MAGIC a comparison against every paper target, all deviations with their
# MAGIC likely effect, the blocked-split robustness results, and which
# MAGIC qualitative findings replicated.
# MAGIC
# MAGIC Everything is re-run rather than read back from CSV, so the report is
# MAGIC guaranteed self-consistent. The whole notebook takes a couple of
# MAGIC minutes on CPU.

# COMMAND ----------

import logging
import sys
from pathlib import Path

REPO_ROOT = Path.cwd()
while not (REPO_ROOT / "configs" / "base.yaml").exists() and REPO_ROOT != REPO_ROOT.parent:
    REPO_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
for noisy in ("acidity_lstm.splits", "acidity_lstm.train", "acidity_lstm.windows"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

from acidity_lstm.config import load_config, on_databricks
from acidity_lstm.experiments import (
    run_fc_baseline,
    run_parametric_study,
    run_refined,
    setup_mlflow,
)
from acidity_lstm.forecast import run_forecast, run_sensitivity
from acidity_lstm.preprocess import clean_acidity, clean_weather
from acidity_lstm.report_phase11 import write_report

print("Running on Databricks:", on_databricks())
cfg = load_config(REPO_ROOT / "configs" / "base.yaml")

weather, _ = clean_weather(cfg)
acidity, _ = clean_acidity(cfg)
mlflow = setup_mlflow(cfg)

# COMMAND ----------

# MAGIC %md ## Phase 6 - parametric study and robustness variants

# COMMAND ----------

table, scenarios = run_parametric_study(
    cfg,
    variants=("paper", "blocked", "common", "fit_train"),
    acidity=acidity,
    weather=weather,
    log_mlflow=False,          # already logged by notebook 01
)
print("scenarios:", len(table))

# COMMAND ----------

# MAGIC %md ## Phase 7 - fully connected baseline

# COMMAND ----------

fc_scenarios = run_fc_baseline(cfg, acidity, weather, mlflow=None)

# COMMAND ----------

# MAGIC %md ## Phase 8 - refined model with the time tag

# COMMAND ----------

refined = run_refined(cfg, acidity, weather, mlflow=None, stations=("BD", "C7"))
refined_blocked = run_refined(cfg, acidity, weather, mlflow=None,
                              stations=("BD",), variant="blocked")

# COMMAND ----------

# MAGIC %md ## Phases 9-10 - forecast and sensitivity

# COMMAND ----------

forecast = run_forecast(cfg, acidity, weather, mlflow=None)
sensitivity, curves, meta = run_sensitivity(cfg, acidity, weather, forecast,
                                            mlflow=None)

# COMMAND ----------

# MAGIC %md ## Assemble the report

# COMMAND ----------

report = write_report(
    cfg,
    phase6_table=table,
    phase6_scenarios=scenarios,
    fc_scenarios=fc_scenarios,
    refined=refined,
    refined_blocked=refined_blocked,
    forecast=forecast,
    sensitivity=sensitivity,
)
print(report)

# COMMAND ----------

if mlflow is not None:
    with mlflow.start_run(run_name="phase11-replication-report"):
        mlflow.log_artifact(str(cfg.reports_dir / "replication_report.md"))
        mlflow.log_text(table.to_csv(index=False), "phase6_all_variants.csv")

# COMMAND ----------

try:
    displayHTML(f"<pre>{report}</pre>")  # noqa: F821  (Databricks builtin)
except NameError:
    pass
