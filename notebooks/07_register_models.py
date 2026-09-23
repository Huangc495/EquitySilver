# Databricks notebook source
# MAGIC %md
# MAGIC # Register the production models
# MAGIC
# MAGIC MLOPS.md phase M2, decision D3. For each station this trains two models:
# MAGIC
# MAGIC - the **champion**: Type B, H = 10, no time tag;
# MAGIC - the **challenger**: the same model with the time tag, which runs in shadow.
# MAGIC
# MAGIC It logs each one as an MLflow pyfunc, with raw daily weather in and acidity
# MAGIC in mg/L out, carrying its own scaler and window code. It then registers each
# MAGIC under its role's alias: in Unity Catalog on Databricks, and in the local
# MAGIC MLflow registry otherwise.
# MAGIC
# MAGIC Pass the commit being trained as the `git_sha` job parameter, since a Git
# MAGIC folder has no `git` binary.

# COMMAND ----------

import os
import sys
from pathlib import Path

REPO_ROOT = Path.cwd()
while not (REPO_ROOT / "configs" / "base.yaml").exists() and REPO_ROOT != REPO_ROOT.parent:
    REPO_ROOT = REPO_ROOT.parent
try:
    import acidity_lstm  # noqa: F401  installed as a wheel (Asset Bundle job)
except ImportError:
    sys.path.insert(0, str(REPO_ROOT / "src"))  # a Git folder or a plain checkout

try:
    os.environ.setdefault("ACIDITY_GIT_SHA", dbutils.widgets.get("git_sha"))  # noqa: F821
except Exception:                                                              # noqa: BLE001
    pass                    # not a job, or no parameter: registry.git_sha falls back

from acidity_lstm.config import load_config
from acidity_lstm.registry import register_models

cfg = load_config(REPO_ROOT / "configs" / "base.yaml")

# COMMAND ----------

summary = register_models(cfg)
print(summary.to_string(index=False))
