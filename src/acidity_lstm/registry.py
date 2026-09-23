"""Train, package and register the production models (MLOPS.md M2; decision D3).

One registered model per station. Its versions are told apart by alias:
`champion` is the untagged Type B model, which serves; `challenger` is the
time-tagged model, which runs in shadow. Both come from the replication's
own training code (`experiments.run_cell`), so a registered champion is the
same model as the matching Table 1 cell.

On Databricks the models go to Unity Catalog as
`<catalog>.<schema>.acidity_<station>`. Locally they go to the MLflow file
registry under the bare name.
"""

from __future__ import annotations

import glob
import hashlib
import json
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import Config, on_databricks
from .experiments import (
    VARIANTS,
    ScenarioResult,
    override,
    prepare_sample_sets,
    run_cell,
    setup_mlflow,
)
from .preprocess import clean_acidity, clean_weather
from .scaling import fit_scaler_for
from .serving import AcidityModel, ModelSpec, scaler_to_dict, signature

log = logging.getLogger(__name__)

ROLES = ("champion", "challenger")
PACKAGE_DIR = Path(__file__).resolve().parent


# --- names and lineage ----------------------------------------------------

def registered_name(cfg: Config, station: str) -> str:
    """`catalog.schema.acidity_bd` on Databricks, `acidity_bd` locally."""
    rcfg = cfg["registry"]
    base = rcfg["model_name"].format(station=station.lower())
    if on_databricks():
        return f"{rcfg['catalog']}.{rcfg['schema']}.{base}"
    return base


def git_sha(repo_root: Path) -> str:
    """The commit being trained. ACIDITY_GIT_SHA wins, since a Databricks
    Git folder has no git binary; the job passes the SHA in."""
    sha = os.environ.get("ACIDITY_GIT_SHA", "").strip()
    if sha:
        return sha
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root,
                             capture_output=True, text=True, timeout=10)
        if out.returncode == 0:
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


def data_fingerprint(cfg: Config) -> str:
    """SHA-256 over the raw input files, so a model names the data it saw."""
    files = sorted(glob.glob(str(cfg.resolve("weather_glob"))))
    files.append(str(cfg.resolve("acidity_excel")))
    h = hashlib.sha256()
    for path in files:
        h.update(Path(path).name.encode("utf-8"))
        h.update(Path(path).read_bytes())
    return h.hexdigest()


# --- training -------------------------------------------------------------

def role_config(cfg: Config, role: str, n_repeats: int | None = None) -> Config:
    """The config a role trains under: the registry's variant plus its time tag."""
    settings = cfg["registry"]["roles"][role]
    rcfg = override(cfg, VARIANTS[cfg["registry"]["variant"]])
    if settings["time_tag"] != "none":
        rcfg = override(rcfg, {"windows": {"time_tag": settings["time_tag"]}})
    if n_repeats is not None:
        rcfg = override(rcfg, {"train": {"n_repeats": int(n_repeats)}})
    return rcfg


def train_role(cfg: Config, acidity, weather, station: str, role: str,
               n_repeats: int | None = None) -> tuple[ScenarioResult, Config]:
    """Train one station's model for one role, exactly as the replication does."""
    settings = cfg["registry"]["roles"][role]
    rcfg = role_config(cfg, role, n_repeats)
    with_tag = settings["time_tag"] != "none"
    sets = prepare_sample_sets(rcfg, acidity, weather,
                               seed=int(rcfg["train"]["base_seed"]),
                               with_time_tag=with_tag)
    sample_set = sets[(station, settings["window_type"])]
    scenario = run_cell(
        rcfg, sample_set, fit_scaler_for(sample_set, rcfg),
        int(settings["hidden_size"]), kind="lstm",
        variant=cfg["registry"]["variant"],
        time_tag=settings["time_tag"] if with_tag else "",
        mlflow=None,
    )
    return scenario, rcfg


def model_spec(scenario: ScenarioResult, rcfg: Config, role: str) -> ModelSpec:
    wcfg, mcfg = rcfg["windows"], rcfg["model"]
    return ModelSpec(
        station=scenario.station,
        role=role,
        window_type=scenario.window_type,
        n_steps=int(wcfg["n_steps"]),
        hidden_size=int(scenario.hidden_size),
        n_features=len(scenario.scaler.feature_names),
        feature_names=tuple(scenario.scaler.feature_names),
        output_activation=mcfg["output_activation"],
        freeze_bias_hh=bool(mcfg.get("freeze_bias_hh", False)),
        with_time_tag=bool(scenario.time_tag),
        time_tag_mode=str(wcfg["time_tag"]),
        time_tag_origin=str(wcfg["time_tag_origin"]),
    )


def input_example(weather: pd.DataFrame, scenario: ScenarioResult, spec: ModelSpec) -> pd.DataFrame:
    """The window behind the station's last sample: complete, so no NaN."""
    days = spec.n_steps * (7 if spec.window_type == "B" else 1)
    end = pd.Timestamp(scenario.meta["date"].max())
    w = weather.loc[:, ["date", "precip_mm", "tmean_c"]]
    return w[(w["date"] > end - pd.Timedelta(days=days)) & (w["date"] <= end)].reset_index(drop=True)


# --- packaging ------------------------------------------------------------

def _copy_package(dest: Path) -> Path:
    """The package's .py files only, so no __pycache__ rides along."""
    target = dest / "acidity_lstm"
    target.mkdir(parents=True)
    for src in PACKAGE_DIR.glob("*.py"):
        shutil.copy2(src, target / src.name)
    return target


def log_model(mlflow, scenario: ScenarioResult, rcfg: Config, spec: ModelSpec,
              example: pd.DataFrame, pip_requirements: list) -> str:
    """Log the pyfunc into the active run. Returns its model URI."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "spec.json").write_text(json.dumps(spec.to_dict(), indent=2), encoding="utf-8")
        (tmp / "scaler.json").write_text(json.dumps(scaler_to_dict(scenario.scaler), indent=2),
                                         encoding="utf-8")
        (tmp / "config.json").write_text(json.dumps(rcfg.raw, indent=2, default=str),
                                         encoding="utf-8")
        torch.save(scenario.best.model.state_dict(), tmp / "weights.pt")
        code = _copy_package(tmp / "code")

        info = mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=AcidityModel(),
            artifacts={
                "spec": str(tmp / "spec.json"),
                "scaler": str(tmp / "scaler.json"),
                "weights": str(tmp / "weights.pt"),
                "config": str(tmp / "config.json"),
            },
            code_paths=[str(code)],
            signature=signature(),
            input_example=example,
            pip_requirements=list(pip_requirements),
        )
    return info.model_uri


def _metrics(scenario: ScenarioResult) -> dict:
    """Per-split R and RMSE in mg/L lead (D-18); normalised MSE is kept too."""
    out = {"n_samples": scenario.n_samples}
    for split, m in scenario.best.evaluation.by_split.items():
        out.update(m.as_dict(split))
    for k, v in scenario.summary.items():
        if isinstance(v, (int, float)) and np.isfinite(v):
            out[f"repeats_{k}"] = v
    return {k: float(v) for k, v in out.items() if v is not None and np.isfinite(v)}


# --- the whole flow -------------------------------------------------------

def register_models(
    cfg: Config,
    acidity: pd.DataFrame | None = None,
    weather: pd.DataFrame | None = None,
    stations=None,
    roles=ROLES,
    n_repeats: int | None = None,
    mlflow=None,
) -> pd.DataFrame:
    """Train, log and register every station x role; set each role's alias.

    Returns one row per registered version.
    """
    if weather is None:
        weather, _ = clean_weather(cfg)
    if acidity is None:
        acidity, _ = clean_acidity(cfg)
    mlflow = mlflow or setup_mlflow(cfg)
    if mlflow is None:
        raise RuntimeError("MLflow is unavailable; cannot register models.")
    if on_databricks():
        mlflow.set_registry_uri("databricks-uc")
    client = mlflow.MlflowClient()

    sha = git_sha(cfg.repo_root)
    fingerprint = data_fingerprint(cfg)
    stations = list(stations or cfg["data"]["stations"])
    rows = []

    for station in stations:
        name = registered_name(cfg, station)
        for role in roles:
            scenario, rcfg = train_role(cfg, acidity, weather, station, role, n_repeats)
            spec = model_spec(scenario, rcfg, role)
            tags = {
                "role": role,
                "station": station,
                "git_sha": sha,
                "data_sha256": fingerprint,
                "recipe": cfg["registry"]["variant"],
                "time_tag": scenario.time_tag or "none",
                "best_seed": str(scenario.best.seed),
            }
            with mlflow.start_run(run_name=f"register-{station}-{role}") as run:
                mlflow.set_tags(tags)
                mlflow.log_params({
                    "station": station, "role": role,
                    "window_type": spec.window_type, "hidden_size": spec.hidden_size,
                    "time_tag": tags["time_tag"], "n_features": spec.n_features,
                })
                mlflow.log_metrics(_metrics(scenario))
                mlflow.log_dict(scaler_to_dict(scenario.scaler), "scaler.json")
                uri = log_model(mlflow, scenario, rcfg, spec,
                                input_example(weather, scenario, spec),
                                cfg["registry"]["pip_requirements"])

            version = mlflow.register_model(uri, name, tags=tags)
            client.set_registered_model_alias(name, role, version.version)
            ev = scenario.best.evaluation.by_split
            rows.append({
                "station": station, "role": role, "name": name,
                "version": int(version.version), "run_id": run.info.run_id,
                "r_all": ev["all"].r, "r_test": ev["test"].r,
                "rmse_test_mgL": ev["test"].rmse_mgL, "best_seed": scenario.best.seed,
                "git_sha": sha,
            })
            log.info("registered %s v%s as @%s", name, version.version, role)
    return pd.DataFrame(rows)
