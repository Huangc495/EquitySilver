# Databricks notebook source
# MAGIC %md
# MAGIC # Train and register (the production pipeline)
# MAGIC
# MAGIC MLOPS.md phase M3; decisions D8 to D10. For each station and role
# MAGIC (champion, then the shadow challenger):
# MAGIC
# MAGIC 1. backtest the candidate recipe from config and the incumbent's recipe
# MAGIC    (rebuilt from its version tags) on the same held-out years;
# MAGIC 2. gate: promote only if the candidate is no worse on out-of-fold RMSE
# MAGIC    in mg/L and R;
# MAGIC 3. on promotion, refit a 5-seed ensemble on every year, register it in
# MAGIC    this environment's schema, and move the role's alias to it.
# MAGIC
# MAGIC Job parameters: `schema` (dev, staging or prod) and `git_sha`. The
# MAGIC Asset Bundle supplies both.

# COMMAND ----------

import os
import sys
from pathlib import Path

REPO_ROOT = Path.cwd()
while not (REPO_ROOT / "configs" / "base.yaml").exists() and REPO_ROOT != REPO_ROOT.parent:
    REPO_ROOT = REPO_ROOT.parent
try:
    import acidity_lstm  # noqa: F401  installed from the bundle's wheel
except ImportError:
    sys.path.insert(0, str(REPO_ROOT / "src"))


def job_parameter(name: str, default: str = "") -> str:
    try:
        return dbutils.widgets.get(name) or default  # noqa: F821
    except Exception:                                # noqa: BLE001
        return os.environ.get(name.upper(), default)


SCHEMA = job_parameter("schema")
os.environ.setdefault("ACIDITY_GIT_SHA", job_parameter("git_sha"))

from acidity_lstm.config import load_config
from acidity_lstm.registry import train_and_register

cfg = load_config(REPO_ROOT / "configs" / "base.yaml")
if cfg["registry"].get("environments") and SCHEMA and SCHEMA not in cfg["registry"]["environments"]:
    raise ValueError(f"schema {SCHEMA!r} is not one of {cfg['registry']['environments']}")
print("acidity_lstm from:", sys.modules["acidity_lstm"].__file__)
print("schema:", SCHEMA or f"{cfg['registry']['schema']} (config fallback)")

# COMMAND ----------

summary = train_and_register(cfg, schema=SCHEMA or None)
print(summary.drop(columns=["run_id"]).round(3).to_string(index=False))
