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


def expand_env(value: str, where: str = "config") -> str:
    """Expand ${VAR} references, failing loudly when a variable is unset.

    Used to keep deployment-specific identifiers -- the workspace username,
    and the Unity Catalog catalog/schema if you choose -- out of the tracked
    config, which is public. `os.path.expandvars` is not used because it
    leaves unset variables in place silently, which would produce a path
    containing a literal "${...}" and a confusing downstream error.
    """
    def substitute(match: re.Match) -> str:
        name = match.group(1)
        resolved = os.environ.get(name)
        if not resolved:
            raise KeyError(
                f"{where} references ${{{name}}}, but that environment "
                f"variable is not set. On Databricks set it on the cluster "
                f"(Compute > Edit > Advanced options > Spark > Environment "
                f"variables), e.g. {name}=you@example.com. Locally, export it "
                f"before running."
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
        return _resolve_against(self.raw["paths"]["reports_dir"], self.repo_root)

    @property
    def mlflow_experiment(self) -> str:
        m = self.raw["mlflow"]
        key = "databricks_experiment" if on_databricks() else "local_experiment"
        return expand_env(m[key], where=f"mlflow.{key}")


def load_config(path: str | Path | None = None) -> Config:
    """Load the YAML config and pick the local or Databricks path set."""
    cfg_path = Path(path) if path is not None else _REPO_ROOT / "configs" / "base.yaml"
    with open(cfg_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    key = "databricks" if on_databricks() else "local"
    paths = dict(raw["paths"][key])
    return Config(raw=raw, paths=paths, repo_root=_REPO_ROOT)
