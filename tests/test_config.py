"""Config loading and local/Databricks path resolution."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from acidity_lstm.config import (
    Config,
    _resolve_against,
    expand_env,
    load_config,
    on_databricks,
)

DATABRICKS_ENV = "DATABRICKS_RUNTIME_VERSION"


@pytest.fixture
def cfg():
    return load_config()


# --- Environment detection ------------------------------------------------

def test_on_databricks_follows_the_environment_variable(monkeypatch):
    monkeypatch.delenv(DATABRICKS_ENV, raising=False)
    assert on_databricks() is False
    monkeypatch.setenv(DATABRICKS_ENV, "16.4.x-cpu-ml-scala2.12")
    assert on_databricks() is True


def test_path_set_switches_with_the_environment(monkeypatch):
    monkeypatch.delenv(DATABRICKS_ENV, raising=False)
    local = load_config()
    assert "data/raw" in str(local.paths["acidity_excel"]).replace("\\", "/")

    monkeypatch.setenv(DATABRICKS_ENV, "16.4")
    cluster = load_config()
    assert str(cluster.paths["acidity_excel"]).startswith("/Volumes/")


def test_mlflow_experiment_switches_with_the_environment(monkeypatch):
    monkeypatch.delenv(DATABRICKS_ENV, raising=False)
    assert load_config().mlflow_experiment == "equity-silver-lstm"

    monkeypatch.setenv(DATABRICKS_ENV, "16.4")
    monkeypatch.setenv("DATABRICKS_USERNAME", "someone@example.com")
    assert load_config().mlflow_experiment.startswith("/Users/")


def test_databricks_experiment_has_no_placeholder(monkeypatch):
    monkeypatch.setenv(DATABRICKS_ENV, "16.4")
    monkeypatch.setenv("DATABRICKS_USERNAME", "someone@example.com")
    experiment = load_config().mlflow_experiment
    assert "<" not in experiment and ">" not in experiment, (
        f"unfilled placeholder in the MLflow experiment path: {experiment}"
    )
    assert experiment == "/Users/someone@example.com/equity-silver-lstm"


def test_no_personal_identifier_is_committed(cfg):
    """The repo is public: deployment identities come from the environment."""
    raw = str(cfg.raw["mlflow"]["databricks_experiment"])
    assert "${DATABRICKS_USERNAME}" in raw
    assert "@" not in raw, f"an address is embedded in the tracked config: {raw}"


# --- Environment-variable expansion ---------------------------------------

def test_expand_env_substitutes_a_set_variable(monkeypatch):
    monkeypatch.setenv("ACIDITY_TEST_VAR", "value")
    assert expand_env("a/${ACIDITY_TEST_VAR}/b") == "a/value/b"


def test_expand_env_handles_several_references(monkeypatch):
    monkeypatch.setenv("ACIDITY_CAT", "teck")
    monkeypatch.setenv("ACIDITY_SCH", "equity")
    out = expand_env("/Volumes/${ACIDITY_CAT}/${ACIDITY_SCH}/raw")
    assert out == "/Volumes/teck/equity/raw"


def test_expand_env_leaves_plain_values_alone():
    assert expand_env("/Volumes/cat/schema/raw") == "/Volumes/cat/schema/raw"
    assert expand_env("data/raw/file.xlsx") == "data/raw/file.xlsx"


def test_expand_env_raises_a_pointed_error_when_unset(monkeypatch):
    """Silent non-expansion would produce a confusing failure much later."""
    monkeypatch.delenv("ACIDITY_MISSING_VAR", raising=False)
    with pytest.raises(KeyError, match="ACIDITY_MISSING_VAR"):
        expand_env("/Users/${ACIDITY_MISSING_VAR}/experiment", where="mlflow")


def test_expand_env_error_names_the_setting(monkeypatch):
    monkeypatch.delenv("ACIDITY_MISSING_VAR", raising=False)
    with pytest.raises(KeyError, match="mlflow.databricks_experiment"):
        expand_env("${ACIDITY_MISSING_VAR}", where="mlflow.databricks_experiment")


def test_empty_variable_is_treated_as_unset(monkeypatch):
    monkeypatch.setenv("ACIDITY_EMPTY_VAR", "")
    with pytest.raises(KeyError, match="ACIDITY_EMPTY_VAR"):
        expand_env("/Users/${ACIDITY_EMPTY_VAR}/x")


def test_missing_username_on_databricks_raises(monkeypatch):
    monkeypatch.setenv(DATABRICKS_ENV, "16.4")
    monkeypatch.delenv("DATABRICKS_USERNAME", raising=False)
    with pytest.raises(KeyError, match="DATABRICKS_USERNAME"):
        _ = load_config().mlflow_experiment


def test_local_experiment_needs_no_environment(monkeypatch):
    """Local runs must not depend on any cluster variable."""
    monkeypatch.delenv(DATABRICKS_ENV, raising=False)
    monkeypatch.delenv("DATABRICKS_USERNAME", raising=False)
    assert load_config().mlflow_experiment == "equity-silver-lstm"


def test_paths_also_support_expansion(monkeypatch, tmp_path):
    monkeypatch.setenv("ACIDITY_CAT", "teck")
    cfg = load_config()
    raw = dict(cfg.paths)
    raw["processed_dir"] = "/Volumes/${ACIDITY_CAT}/schema/processed"
    patched = Config(raw=cfg.raw, paths=raw, repo_root=cfg.repo_root)
    assert patched.resolve("processed_dir").as_posix() == "/Volumes/teck/schema/processed"


# --- Path resolution ------------------------------------------------------

def test_posix_absolute_paths_are_never_rerooted():
    """Regression: on Windows, Path("/Volumes/x").is_absolute() is False.

    Without an explicit leading-slash check, resolving the cluster path set on
    a Windows machine silently re-rooted it to C:\\Volumes\\x.
    """
    root = Path("C:/repo") if sys.platform == "win32" else Path("/repo")
    out = _resolve_against("/Volumes/cat/schema/raw/file.xlsx", root)
    assert out.as_posix() == "/Volumes/cat/schema/raw/file.xlsx"
    assert "repo" not in out.as_posix()


def test_relative_paths_resolve_against_the_repo_root():
    root = Path("C:/repo") if sys.platform == "win32" else Path("/repo")
    out = _resolve_against("data/raw/file.xlsx", root)
    assert out == root / "data/raw/file.xlsx"


def test_cluster_paths_resolve_unchanged(monkeypatch):
    monkeypatch.setenv(DATABRICKS_ENV, "16.4")
    cfg = load_config()
    for key in ("weather_glob", "acidity_excel", "processed_dir"):
        resolved = cfg.resolve(key).as_posix()
        assert resolved == str(cfg.paths[key]), key
        assert resolved.startswith("/Volumes/"), key


def test_local_paths_are_absolute_and_point_at_real_data(monkeypatch):
    monkeypatch.delenv(DATABRICKS_ENV, raising=False)
    cfg = load_config()
    assert cfg.resolve("acidity_excel").is_absolute()
    assert cfg.resolve("acidity_excel").exists()
    assert cfg.reports_dir.is_absolute()


def test_resolution_is_independent_of_the_working_directory(monkeypatch, tmp_path):
    monkeypatch.delenv(DATABRICKS_ENV, raising=False)
    before = load_config().resolve("acidity_excel")
    monkeypatch.chdir(tmp_path)
    assert load_config().resolve("acidity_excel") == before


# --- Config access --------------------------------------------------------

def test_sections_are_reachable(cfg):
    for section in ("data", "windows", "scaling", "model", "train", "split",
                    "forecast", "sensitivity", "mlflow", "paths"):
        assert section in cfg.raw, section
    assert cfg["train"]["base_seed"] == 42
    assert cfg.get("nonexistent") is None
    assert cfg.get("nonexistent", "fallback") == "fallback"


def test_unknown_path_key_raises(cfg):
    with pytest.raises(KeyError):
        cfg.resolve("not_a_path")
