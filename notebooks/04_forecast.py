# Databricks notebook source
# MAGIC %md
# MAGIC # Phases 9-10 - Forecast and sensitivity
# MAGIC
# MAGIC Trains the refined model on 1999-2014 BD samples, forecasts 2015-2016
# MAGIC from the actual weather (Fig. 11), then re-predicts that period under
# MAGIC +-20% precipitation and temperature (Fig. 12).
# MAGIC
# MAGIC Writes `reports/forecast_sensitivity.md`.
# MAGIC
# MAGIC Both phases share one trained model, so they live in one notebook;
# MAGIC `05_sensitivity.py` re-runs Phase 10 alone.

# COMMAND ----------

import logging
import sys
from pathlib import Path

REPO_ROOT = Path.cwd()
while not (REPO_ROOT / "configs" / "base.yaml").exists() and REPO_ROOT != REPO_ROOT.parent:
    REPO_ROOT = REPO_ROOT.parent
try:
    import acidity_lstm  # noqa: F401  installed as a wheel (Asset Bundle job)
except ImportError:
    sys.path.insert(0, str(REPO_ROOT / "src"))  # a Git folder or a plain checkout

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logging.getLogger("acidity_lstm.splits").setLevel(logging.WARNING)

from acidity_lstm.config import load_config, on_databricks
from acidity_lstm.experiments import setup_mlflow
from acidity_lstm.forecast import run_forecast, run_sensitivity
from acidity_lstm.preprocess import clean_acidity, clean_weather
from acidity_lstm.report_phase910 import (
    make_forecast_figures,
    make_sensitivity_figure,
    write_report,
)

print("Running on Databricks:", on_databricks())
cfg = load_config(REPO_ROOT / "configs" / "base.yaml")

# COMMAND ----------

weather, _ = clean_weather(cfg)
acidity, _ = clean_acidity(cfg)
mlflow = setup_mlflow(cfg)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Phase 9 - forecast
# MAGIC
# MAGIC The scaler is fitted on the training period only, and best-of-5 is
# MAGIC selected on train+validation MSE so the forecast period never informs
# MAGIC the choice.

# COMMAND ----------

fc = run_forecast(cfg, acidity, weather, mlflow=mlflow)
print("split counts:", fc.split_counts)
print(f"train+val MSE {fc.train_val_mse():.3f} | forecast MSE {fc.predict_mse():.3f}")
print("forecast MSE spread:", {k: round(v, 3) for k, v in fc.spread().items()
                               if k != "values"})

# COMMAND ----------

# MAGIC %md ## Phase 10 - sensitivity

# COMMAND ----------

table, curves, meta = run_sensitivity(cfg, acidity, weather, fc, mlflow=mlflow)
table.round(1)

# COMMAND ----------

# MAGIC %md ## Figures and report

# COMMAND ----------

figs = make_forecast_figures(cfg, fc)
figs.update(make_sensitivity_figure(cfg, curves, meta, table))
report = write_report(cfg, fc, table, curves, figs)
print(report)

# COMMAND ----------

table.to_csv(cfg.reports_dir / "sensitivity_raw.csv", index=False)

if mlflow is not None:
    with mlflow.start_run(run_name="phase910-summary"):
        for path in figs.values():
            mlflow.log_artifact(str(path), artifact_path="figures")
        mlflow.log_artifact(str(cfg.reports_dir / "forecast_sensitivity.md"))
        mlflow.log_text(table.to_csv(index=False), "sensitivity_raw.csv")

# COMMAND ----------

try:
    displayHTML(f"<pre>{report}</pre>")  # noqa: F821  (Databricks builtin)
except NameError:
    pass
