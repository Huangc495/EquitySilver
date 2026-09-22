"""Configuration loading and local/Databricks path resolution."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]


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
        value = Path(str(self.paths[path_key]))
        if value.is_absolute():
            return value
        return self.repo_root / value

    @property
    def reports_dir(self) -> Path:
        value = Path(str(self.raw["paths"]["reports_dir"]))
        return value if value.is_absolute() else self.repo_root / value

    @property
    def mlflow_experiment(self) -> str:
        m = self.raw["mlflow"]
        return m["databricks_experiment"] if on_databricks() else m["local_experiment"]


def load_config(path: str | Path | None = None) -> Config:
    """Load the YAML config and pick the local or Databricks path set."""
    cfg_path = Path(path) if path is not None else _REPO_ROOT / "configs" / "base.yaml"
    with open(cfg_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    key = "databricks" if on_databricks() else "local"
    paths = dict(raw["paths"][key])
    return Config(raw=raw, paths=paths, repo_root=_REPO_ROOT)
