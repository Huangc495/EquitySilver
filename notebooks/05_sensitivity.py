# Databricks notebook source
# MAGIC %md
# MAGIC # Phase 10 - Sensitivity (standalone)
# MAGIC
# MAGIC Phase 10 needs the Phase 9 model, so this notebook re-runs the forecast
# MAGIC first. It is deterministic given the config seed, so it reproduces the
# MAGIC same model `04_forecast.py` used. Run `04_forecast.py` for the full
# MAGIC forecast write-up; this notebook is for exploring the sensitivity grid
# MAGIC on its own.

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
from acidity_lstm.forecast import run_forecast, run_sensitivity
from acidity_lstm.preprocess import clean_acidity, clean_weather
from acidity_lstm.report_phase910 import make_sensitivity_figure

print("Running on Databricks:", on_databricks())
cfg = load_config(REPO_ROOT / "configs" / "base.yaml")

weather, _ = clean_weather(cfg)
acidity, _ = clean_acidity(cfg)

# COMMAND ----------

# MAGIC %md ## Rebuild the Phase 9 model

# COMMAND ----------

fc = run_forecast(cfg, acidity, weather, mlflow=None)
print(f"forecast MSE {fc.predict_mse():.3f} (selected seed {fc.best.seed})")

# COMMAND ----------

# MAGIC %md ## Sensitivity grid

# COMMAND ----------

table, curves, meta = run_sensitivity(cfg, acidity, weather, fc, mlflow=None)
table.round(2)

# COMMAND ----------

figs = make_sensitivity_figure(cfg, curves, meta, table)
print("wrote", figs["fig12"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Direction check
# MAGIC
# MAGIC The paper expects: more precipitation lowers acidity, a larger
# MAGIC temperature swing raises it, and temperature is the more sensitive
# MAGIC input.

# COMMAND ----------

for variable in cfg["sensitivity"]["variables"]:
    sub = table[table["variable"] == variable]
    swing = sub["mean_change_pct"].abs().sum()
    print(f"{variable:7s} total swing {swing:5.2f}%  "
          + "  ".join(f"x{r.factor:g}: {r.mean_change_mgL:+7.0f} mg/L"
                      for r in sub.itertuples()))
