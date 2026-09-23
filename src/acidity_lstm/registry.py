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

def registered_name(cfg: Config, station: str, schema: str | None = None) -> str:
    """`catalog.schema.acidity_bd` on Databricks, `acidity_bd` locally.

    `schema` is the environment (dev, staging or prod; MLOPS.md D8); the
    config's `registry.schema` is the fallback.
    """
    rcfg = cfg["registry"]
    base = rcfg["model_name"].format(station=station.lower())
    if on_databricks():
        return f"{rcfg['catalog']}.{schema or rcfg['schema']}.{base}"
    return base if not schema else f"{schema}_{base}"


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


def build_spec(rcfg: Config, station: str, role: str, window_type: str, hidden_size: int,
               scaler, with_time_tag: bool, n_members: int = 1) -> ModelSpec:
    wcfg, mcfg = rcfg["windows"], rcfg["model"]
    return ModelSpec(
        station=station,
        role=role,
        window_type=window_type,
        n_steps=int(wcfg["n_steps"]),
        hidden_size=int(hidden_size),
        n_features=len(scaler.feature_names),
        feature_names=tuple(scaler.feature_names),
        output_activation=mcfg["output_activation"],
        freeze_bias_hh=bool(mcfg.get("freeze_bias_hh", False)),
        with_time_tag=bool(with_time_tag),
        time_tag_mode=str(wcfg["time_tag"]),
        time_tag_origin=str(wcfg["time_tag_origin"]),
        n_members=int(n_members),
    )


def model_spec(scenario: ScenarioResult, rcfg: Config, role: str) -> ModelSpec:
    return build_spec(rcfg, scenario.station, role, scenario.window_type,
                      scenario.hidden_size, scenario.scaler, bool(scenario.time_tag))


def input_example(weather: pd.DataFrame, meta: pd.DataFrame, spec: ModelSpec) -> pd.DataFrame:
    """The window behind the station's last sample: complete, so no NaN."""
    days = spec.n_steps * (7 if spec.window_type == "B" else 1)
    end = pd.Timestamp(meta["date"].max())
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


def log_model(mlflow, members, scaler, rcfg: Config, spec: ModelSpec,
              example: pd.DataFrame, pip_requirements: list) -> str:
    """Log the pyfunc into the active run. Returns its model URI.

    `members` is the list of networks the model averages (one or more).
    """
    members = list(members) if isinstance(members, (list, tuple)) else [members]
    if len(members) != spec.n_members:
        raise ValueError(f"{len(members)} network(s) for a spec of {spec.n_members}.")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "spec.json").write_text(json.dumps(spec.to_dict(), indent=2), encoding="utf-8")
        (tmp / "scaler.json").write_text(json.dumps(scaler_to_dict(scaler), indent=2),
                                         encoding="utf-8")
        (tmp / "config.json").write_text(json.dumps(rcfg.raw, indent=2, default=str),
                                         encoding="utf-8")
        torch.save([m.state_dict() for m in members], tmp / "weights.pt")
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
                uri = log_model(mlflow, [scenario.best.model], scenario.scaler, rcfg, spec,
                                input_example(weather, scenario.meta, spec),
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


# --- the production flow (MLOPS.md M3; D9, D10) ----------------------------

def incumbent(client, name: str, role: str):
    """The version currently holding `role`'s alias, or None."""
    try:
        return client.get_model_version_by_alias(name, role)
    except Exception as exc:                      # noqa: BLE001
        if "RESOURCE_DOES_NOT_EXIST" in str(exc) or "not found" in str(exc).lower() \
                or "does not exist" in str(exc).lower():
            return None
        raise


def train_and_register(
    cfg: Config,
    acidity: pd.DataFrame | None = None,
    weather: pd.DataFrame | None = None,
    stations=None,
    roles=ROLES,
    schema: str | None = None,
    mlflow=None,
) -> pd.DataFrame:
    """Backtest, gate, refit and register every station x role.

    For each slot the candidate recipe (from config) and the incumbent's
    recipe (from its version tags) are backtested on the same folds. If the
    gate passes, the candidate is refit on every year, registered, and given
    the role's alias. Otherwise the alias stays where it is. Either way one
    MLflow run records both backtests and the decision.

    The challenger slot competes only with the incumbent challenger; nothing
    here moves a model from challenger to champion (D3).
    """
    from .backtest import backtest, candidate_recipe, gate, recipe_from_tags, refit

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
    n_seeds = int(cfg["gate"]["n_seeds"])
    rows = []

    for station in list(stations or cfg["data"]["stations"]):
        name = registered_name(cfg, station, schema)
        for role in roles:
            cand = candidate_recipe(cfg, role)
            inc_version = incumbent(client, name, role)
            inc = recipe_from_tags(inc_version.tags, n_seeds) if inc_version else None

            bt_c = backtest(cfg, acidity, weather, station, cand)
            bt_i = None
            if inc is not None:
                # Training is deterministic, so an unchanged recipe scores the same.
                bt_i = bt_c if inc == cand else backtest(cfg, acidity, weather, station, inc)
            decision = gate(bt_c, bt_i, cfg)

            tags = {
                "role": role, "station": station, "git_sha": sha,
                "data_sha256": fingerprint, "recipe": "production",
                "recipe_json": cand.to_json(), "time_tag": cand.time_tag,
                "gate": "promoted" if decision.promote else "rejected",
                "gate_reason": decision.reason,
                "incumbent_version": inc_version.version if inc_version else "none",
            }
            row = {"station": station, "role": role, "name": name,
                   "promoted": decision.promote, "reason": decision.reason,
                   "cand_r": bt_c.r, "cand_rmse_mgL": bt_c.rmse_mgL,
                   "inc_r": bt_i.r if bt_i else np.nan,
                   "inc_rmse_mgL": bt_i.rmse_mgL if bt_i else np.nan,
                   "version": None, "epochs": bt_c.median_epochs}

            with mlflow.start_run(run_name=f"train-{station}-{role}") as run:
                mlflow.set_tags(tags)
                mlflow.log_params({"station": station, "role": role,
                                   "candidate": cand.to_json(),
                                   "incumbent": inc.to_json() if inc else "none",
                                   "folds": json.dumps(bt_c.folds)})
                metrics = bt_c.metrics("candidate")
                if bt_i is not None:
                    metrics.update(bt_i.metrics("incumbent"))
                mlflow.log_metrics({k: float(v) for k, v in metrics.items() if np.isfinite(v)})
                mlflow.log_dict({"candidate": bt_c.per_fold,
                                 "incumbent": bt_i.per_fold if bt_i else None},
                                "backtest_folds.json")

                if decision.promote:
                    members, scaler, s = refit(cfg, acidity, weather, station, cand,
                                               bt_c.median_epochs)
                    rcfg = override(cfg, {"windows": {"time_tag": cand.time_tag}}) \
                        if cand.with_time_tag else cfg
                    spec = build_spec(rcfg, station, role, cand.window_type, cand.hidden_size,
                                      scaler, cand.with_time_tag, n_members=len(members))
                    mlflow.log_dict(scaler_to_dict(scaler), "scaler.json")
                    uri = log_model(mlflow, members, scaler, rcfg, spec,
                                    input_example(weather, s.meta, spec),
                                    cfg["registry"]["pip_requirements"])

            if decision.promote:
                version = mlflow.register_model(uri, name, tags={
                    **tags, "oof_r": f"{bt_c.r:.4f}", "oof_rmse_mgL": f"{bt_c.rmse_mgL:.1f}"})
                client.set_registered_model_alias(name, role, version.version)
                row["version"] = int(version.version)
                log.info("%s @%s -> v%s (%s)", name, role, version.version, decision.reason)
            else:
                log.info("%s @%s unchanged (%s)", name, role, decision.reason)
            row["run_id"] = run.info.run_id
            rows.append(row)
    return pd.DataFrame(rows)
