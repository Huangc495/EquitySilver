# Databricks notebook source
# MAGIC %md
# MAGIC # Cluster preflight
# MAGIC
# MAGIC Run this **first** on a new cluster, before `00_data_audit`. It checks
# MAGIC the three things that otherwise fail partway through a notebook with a
# MAGIC confusing error:
# MAGIC
# MAGIC 1. the environment resolves (`DATABRICKS_USERNAME`, MLflow experiment);
# MAGIC 2. the raw data is where the config expects it in the Volume;
# MAGIC 3. the installed packages match `requirements.txt`.
# MAGIC
# MAGIC It changes nothing. Safe to re-run at any time.

# COMMAND ----------

import sys
from pathlib import Path

REPO_ROOT = Path.cwd()
while not (REPO_ROOT / "configs" / "base.yaml").exists() and REPO_ROOT != REPO_ROOT.parent:
    REPO_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from acidity_lstm.config import load_config
from acidity_lstm.preflight import environment_report, run_preflight

cfg = load_config(REPO_ROOT / "configs" / "base.yaml")

# COMMAND ----------

# MAGIC %md ## Checks

# COMMAND ----------

passed, table = run_preflight(cfg)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Installed versions
# MAGIC
# MAGIC Paste this into `requirements.txt` to pin against the cluster this
# MAGIC actually runs on (DEVIATIONS.md D-01). The runtime's own versions are
# MAGIC the ones to trust; any WARN above just means the file disagrees.

# COMMAND ----------

print(environment_report())

# COMMAND ----------

# MAGIC %md ## Gate
# MAGIC
# MAGIC Fails the notebook if anything is wrong, so a scheduled job stops here
# MAGIC rather than part-way through the pipeline.

# COMMAND ----------

if not passed:
    raise SystemExit(
        "Preflight failed. See the FAIL rows above.\n"
        "Common causes: DATABRICKS_USERNAME not set as a cluster environment "
        "variable, or the raw data not yet uploaded to the Volume. "
        "See docs/README.md."
    )
print("Preflight OK - run 00_data_audit next.")
