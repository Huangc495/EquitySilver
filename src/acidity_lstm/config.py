"""Configuration loading and local/Databricks path resolution."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]

# ${VAR} references in config values, expanded from the environment.
_ENV_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def databricks_current_user() -> str:
    """The signed-in workspace user, asked of Databricks directly.

    Serverless compute has no cluster environment variables, so
    `DATABRICKS_USERNAME` cannot be set there the way a classic cluster
    allows. Asking the platform removes that setup step everywhere. Returns
    "" off-cluster or when no source answers.
    """
    try:
        from pyspark.sql import SparkSession

        spark = SparkSession.getActiveSession()
        if spark is not None:
            user = spark.sql("SELECT current_user()").collect()[0][0]
            if user:
                return str(user)
    except Exception:                                # noqa: BLE001
        pass

    try:
        from databricks.sdk.runtime import dbutils

        ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
        user = ctx.tags().apply("user")
        if user:
            return str(user)
    except Exception:                                # noqa: BLE001
        pass
    return ""


# Variables that can be resolved from the platform when unset in the
# environment, so a missing cluster setting is not a hard stop.
_ENV_FALLBACKS = {"DATABRICKS_USERNAME": databricks_current_user}


def expand_env(value: str, where: str = "config") -> str:
    """Expand ${VAR} references, failing loudly when a variable is unset.

    Used to keep deployment-specific identifiers -- the workspace username,
    and the Unity Catalog catalog/schema if you choose -- out of the tracked
    config, which is public. `os.path.expandvars` is not used because it
    leaves unset variables in place silently, which would produce a path
    containing a literal "${...}" and a confusing downstream error.

    A few variables have a platform fallback (see `_ENV_FALLBACKS`); the
    environment still wins when it is set.
    """
    def substitute(match: re.Match) -> str:
        name = match.group(1)
        resolved = os.environ.get(name)
        if not resolved and name in _ENV_FALLBACKS:
            resolved = _ENV_FALLBACKS[name]()
        if not resolved:
            raise KeyError(
                f"{where} references ${{{name}}}, but that environment "
                f"variable is not set and could not be resolved from the "
                f"platform. On a classic cluster set it under Compute > Edit "
                f"> Advanced options > Spark > Environment variables, e.g. "
                f"{name}=you@example.com. In a notebook you can also set it "
                f"directly: os.environ['{name}'] = 'you@example.com'."
            )
        return resolved

    return _ENV_REF.sub(substitute, str(value))


def _resolve_against(value, root: Path) -> Path:
    """Resolve a configured path, treating a leading "/" as absolute.

    The cluster paths (/Volumes/...) are POSIX-absolute, but Windows pathlib
    does not consider a bare leading slash absolute. Without this check,
    resolving the Databricks path set on a Windows machine would silently
    re-root it against the repo drive (C:\\Volumes\\...) instead of leaving it
    alone -- wrong, and easy to miss because the cluster itself is fine.
    """
    text = expand_env(value, where="path")
    path = Path(text)
    if text.startswith("/") or path.is_absolute():
        return path
    return root / path


def on_databricks() -> bool:
    """True when running inside a Databricks cluster."""
    return "DATABRICKS_RUNTIME_VERSION" in os.environ


@dataclass(frozen=True)
class Config:
    """Parsed `configs/base.yaml` with the active path set already resolved."""

    raw: dict[str, Any]
    paths: dict[str, Any]
    repo_root: Path

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)

    def resolve(self, path_key: str) -> Path:
        """Resolve a key of the active path set to an absolute path.

        Local paths are taken relative to the repo root so that notebooks and
        tests behave the same regardless of the working directory. Databricks
        paths are already absolute (/Volumes/...).
        """
        return _resolve_against(self.paths[path_key], self.repo_root)

    @property
    def reports_dir(self) -> Path:
        """Where generated reports go: the repo locally, a volume on a cluster.

        On Databricks the repo is a Git folder, so writing reports into it
        would leave tracked files modified and make the next pull conflict.
        """
        return self.resolve("reports_dir")

    @property
    def mlflow_experiment(self) -> str:
        m = self.raw["mlflow"]
        key = "databricks_experiment" if on_databricks() else "local_experiment"
        return expand_env(m[key], where=f"mlflow.{key}")


def load_config(path: str | Path | None = None) -> Config:
    """Load the YAML config and pick the local or Databricks path set.

    The repo root is taken from the config's own location
    (`<root>/configs/base.yaml`), not from where this module is installed.
    The two agree in a checkout or an editable install, but an installed
    wheel lives in site-packages, where no `requirements.txt`,
    `DEVIATIONS.md` or `reports/` exists. Pass the path explicitly there, as
    every notebook does.
    """
    cfg_path = Path(path) if path is not None else _REPO_ROOT / "configs" / "base.yaml"
    with open(cfg_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    key = "databricks" if on_databricks() else "local"
    paths = dict(raw["paths"][key])
    return Config(raw=raw, paths=paths, repo_root=cfg_path.resolve().parents[1])
