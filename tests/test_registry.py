"""Registering the production models (MLOPS.md M2; decision D3)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from acidity_lstm.config import load_config
from acidity_lstm.registry import (
    ROLES,
    data_fingerprint,
    git_sha,
    register_models,
    registered_name,
    role_config,
)

DATABRICKS_ENV = "DATABRICKS_RUNTIME_VERSION"


@pytest.fixture
def cfg():
    return load_config()


# --- names, roles and lineage ---------------------------------------------

def test_names_are_three_level_only_on_databricks(cfg, monkeypatch):
    monkeypatch.delenv(DATABRICKS_ENV, raising=False)
    assert registered_name(cfg, "BD") == "acidity_bd"
    monkeypatch.setenv(DATABRICKS_ENV, "client.4.10")
    assert registered_name(cfg, "C7") == "equity_silver_databricks_mlops.default.acidity_c7"


def test_the_roles_are_decision_d3(cfg):
    """Champion: untagged Type B H=10. Challenger: the same with the time tag."""
    roles = cfg["registry"]["roles"]
    assert set(roles) == set(ROLES)
    assert roles["champion"]["time_tag"] == "none"
    assert roles["challenger"]["time_tag"] != "none"
    for r in ROLES:
        assert (roles[r]["window_type"], roles[r]["hidden_size"]) == ("B", 10)
    assert role_config(cfg, "challenger")["windows"]["time_tag"] == roles["challenger"]["time_tag"]


def test_git_sha_prefers_the_environment(cfg, monkeypatch):
    monkeypatch.setenv("ACIDITY_GIT_SHA", "abc123")
    assert git_sha(cfg.repo_root) == "abc123"
    monkeypatch.delenv("ACIDITY_GIT_SHA")
    sha = git_sha(cfg.repo_root)
    assert sha == "unknown" or len(sha) == 40


def test_model_requirements_pin_cpu_torch(cfg):
    reqs = cfg["registry"]["pip_requirements"]
    assert any(r.startswith("torch==") and r.endswith("+cpu") for r in reqs)
    assert any("download.pytorch.org/whl/cpu" in r for r in reqs)


@pytest.mark.integration
def test_data_fingerprint_is_stable(cfg):
    a, b = data_fingerprint(cfg), data_fingerprint(cfg)
    assert a == b and len(a) == 64


# --- the whole flow, into a throwaway local registry -----------------------

@pytest.fixture(scope="module")
def registered(tmp_path_factory):
    """Register BD's champion and challenger (one repeat each) locally."""
    import mlflow

    cfg = load_config()
    root = tmp_path_factory.mktemp("registry")
    previous = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(root.as_uri())
    mlflow.set_experiment("registry-test")
    try:
        summary = register_models(cfg, stations=["BD"], n_repeats=1, mlflow=mlflow)
        yield cfg, mlflow, summary
    finally:
        mlflow.set_tracking_uri(previous)


@pytest.mark.integration
def test_each_role_gets_a_version_and_its_alias(registered):
    _, mlflow, summary = registered
    assert list(summary["role"]) == list(ROLES)
    client = mlflow.MlflowClient()
    for row in summary.itertuples():
        mv = client.get_model_version_by_alias(row.name, row.role)
        assert int(mv.version) == row.version
        assert mv.tags["role"] == row.role
        assert mv.tags["data_sha256"]
    assert sorted(summary["version"]) == [1, 2]


@pytest.mark.integration
def test_registered_champion_serves_what_training_predicted(registered):
    """No training/serving skew: the loaded model, fed raw weather, returns the
    training code's own predictions at every sample date."""
    from acidity_lstm.preprocess import clean_acidity, clean_weather
    from acidity_lstm.registry import train_role

    cfg, mlflow, summary = registered
    weather, _ = clean_weather(cfg)
    acidity, _ = clean_acidity(cfg)
    for role in ROLES:
        scenario, _ = train_role(cfg, acidity, weather, "BD", role, n_repeats=1)
        name = summary.loc[summary["role"] == role, "name"].iloc[0]
        model = mlflow.pyfunc.load_model(f"models:/{name}@{role}")

        out = model.predict(weather).set_index("date")
        served = out.loc[pd.DatetimeIndex(scenario.meta["date"]), "acidity_mgL"].to_numpy()
        np.testing.assert_allclose(served, scenario.best.evaluation.y_pred_mgL, rtol=1e-6)
        assert out.loc[pd.DatetimeIndex(scenario.meta["date"]), "window_complete"].all()


@pytest.mark.integration
def test_registered_metrics_lead_with_r_and_rmse_in_mgl(registered):
    """Promotion compares R and RMSE in mg/L, never normalised MSE (D-18)."""
    _, mlflow, summary = registered
    run = mlflow.get_run(summary["run_id"].iloc[0])
    for split in ("train", "val", "test", "all"):
        assert f"{split}_r" in run.data.metrics
        assert f"{split}_rmse_mgL" in run.data.metrics
    assert run.data.tags["git_sha"]
