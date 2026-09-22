"""Experiment grids, repeats and MLflow logging (CLAUDE.md Phase 6).

One MLflow run per scenario (station x input type x hidden size x variant),
with one nested run per repeat carrying that repeat's seed and metrics.
"""

from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config, on_databricks
from .evaluate import Evaluation
from .preprocess import clean_acidity, clean_weather
from .scaling import Scaler, fit_scaler_for
from .splits import SPLIT_NAMES, assign_splits, eligible_years_from, realised_counts
from .train import TrainResult, best_of, summarise_repeats, train_repeats
from .windows import build_sample_sets, make_weather_arrays

log = logging.getLogger(__name__)

# Table 1 of Ma et al. (2021): best-of-5 MSE in normalised units.
PAPER_TABLE1 = {
    ("BD", "A"): {5: 0.79, 10: 0.76, 20: 0.76},
    ("BD", "B"): {5: 0.59, 10: 0.54, 20: 0.54},
    ("C7", "A"): {5: 0.85, 10: 0.83, 20: 0.81},
    ("C7", "B"): {5: 0.77, 10: 0.74, 20: 0.73},
}

# Paper Fig. 6 correlation at Type B, H = 10.
PAPER_FIG6_R = {"BD": 0.70, "C7": 0.51}

# Config overrides defining each variant of the grid.
VARIANTS: dict[str, dict] = {
    "paper": {},
    "blocked": {"split": {"method": "blocked"}},
    "common": {"windows": {"common_sample_set": True}},
    "fit_train": {"scaling": {"fit_on": "train"}},
}

VARIANT_LABELS = {
    "paper": "Paper settings (random split, scaler on all samples)",
    "blocked": "Blocked split (whole years held out)",
    "common": "Common sample set (Type A restricted to Type B survivors)",
    "fit_train": "Scaler fitted on training samples only",
}


def override(cfg: Config, updates: dict) -> Config:
    """A copy of `cfg` with nested sections shallow-updated."""
    raw = copy.deepcopy(cfg.raw)
    for section, section_updates in updates.items():
        raw[section].update(section_updates)
    return Config(raw=raw, paths=cfg.paths, repo_root=cfg.repo_root)


@dataclass
class ScenarioResult:
    """One (station, type, hidden size) cell of the grid."""

    variant: str
    station: str
    window_type: str
    hidden_size: int
    n_samples: int
    split_counts: dict
    summary: dict
    best: TrainResult
    results: list = field(default_factory=list)
    scaler: Scaler | None = None
    seconds: float = 0.0
    # Row-aligned metadata (station, date, acidity_mgL, day_number, split),
    # kept so the time-series figures can be drawn from a scenario alone.
    meta: pd.DataFrame | None = None
    kind: str = "lstm"              # "lstm" | "fc"
    time_tag: str = ""              # "", "per_step" or "constant"

    @property
    def paper_mse(self) -> float:
        """Table 1's value, which only applies to the plain LSTM grid."""
        if self.kind != "lstm" or self.time_tag:
            return float("nan")
        return PAPER_TABLE1.get((self.station, self.window_type), {}).get(
            self.hidden_size, float("nan")
        )

    def row(self) -> dict:
        b = self.best.evaluation
        return {
            "variant": self.variant,
            "kind": self.kind,
            "time_tag": self.time_tag,
            "station": self.station,
            "type": self.window_type,
            "hidden": self.hidden_size,
            "n": self.n_samples,
            "n_train": self.split_counts["train"],
            "n_val": self.split_counts["val"],
            "n_test": self.split_counts["test"],
            "best_mse": self.summary["best_all_mse"],
            "paper_mse": self.paper_mse,
            "diff": self.summary["best_all_mse"] - self.paper_mse,
            "mse_mean": self.summary["all_mse_mean"],
            "mse_sd": self.summary["all_mse_sd"],
            "r_all": b.by_split["all"].r,
            "mse_train": b.by_split["train"].mse,
            "r_train": b.by_split["train"].r,
            "mse_val": b.by_split["val"].mse,
            "r_val": b.by_split["val"].r,
            "mse_test": b.by_split["test"].mse,
            "r_test": b.by_split["test"].r,
            "rmse_mgL": b.by_split["all"].rmse_mgL,
            "best_seed": self.summary["best_seed"],
            "epochs_mean": self.summary["epochs_mean"],
            "seconds": round(self.seconds, 2),
        }


# --- MLflow ---------------------------------------------------------------

def setup_mlflow(cfg: Config):
    """Point MLflow at the local or Databricks experiment. Returns the module.

    Returns None if MLflow is unavailable, so a missing tracking server never
    blocks an experiment.
    """
    try:
        import mlflow
    except ImportError:
        log.warning("MLflow is not installed; runs will not be logged.")
        return None

    try:
        if not on_databricks():
            mlflow.set_tracking_uri(cfg["mlflow"]["local_tracking_uri"])
        mlflow.set_experiment(cfg.mlflow_experiment)
        log.info("MLflow experiment: %s", cfg.mlflow_experiment)
        return mlflow
    except Exception as exc:                      # noqa: BLE001
        log.warning("Could not set up MLflow (%s); runs will not be logged.", exc)
        return None


def _flat_config_params(cfg: Config) -> dict:
    """Flatten the config sections that affect results, for logging."""
    out = {}
    for section in ("windows", "scaling", "model", "train", "split"):
        for key, value in cfg[section].items():
            out[f"{section}.{key}"] = str(value)
    return out


def _log_scenario(mlflow, scenario: ScenarioResult, cfg: Config) -> None:
    """Log one scenario and its repeats (CLAUDE.md rule 7)."""
    if mlflow is None:
        return

    name = (f"{scenario.variant}-{scenario.kind}-{scenario.station}"
            f"-{scenario.window_type}-H{scenario.hidden_size}"
            + (f"-tag_{scenario.time_tag}" if scenario.time_tag else ""))
    with mlflow.start_run(run_name=name):
        mlflow.log_params(
            {
                "variant": scenario.variant,
                "kind": scenario.kind,
                "time_tag": scenario.time_tag or "none",
                "station": scenario.station,
                "window_type": scenario.window_type,
                "hidden_size": scenario.hidden_size,
                "n_features": scenario.best.model.n_features
                if hasattr(scenario.best.model, "n_features") else "",
                "n_parameters": scenario.best.n_parameters,
                **_flat_config_params(cfg),
            }
        )
        mlflow.log_metrics(
            {
                "n_samples": scenario.n_samples,
                **{f"n_{k}": v for k, v in scenario.split_counts.items()},
                "paper_mse": scenario.paper_mse,
                **{k: v for k, v in scenario.summary.items()
                   if isinstance(v, (int, float)) and np.isfinite(v)},
            }
        )
        if scenario.scaler is not None:
            params = scenario.scaler.params
            mlflow.log_params({k: v for k, v in params.items() if isinstance(v, str)})
            mlflow.log_metrics(
                {k: float(v) for k, v in params.items() if not isinstance(v, str)}
            )
            # Also as an artifact, so the exact transform can be recovered.
            mlflow.log_dict(
                {
                    **{k: (v if isinstance(v, str) else float(v))
                       for k, v in params.items()},
                    "feature_names": list(scenario.scaler.feature_names),
                    "x_mean": [float(v) for v in scenario.scaler.x_mean],
                    "x_std": [float(v) for v in scenario.scaler.x_std],
                },
                "scaler.json",
            )

        for r in scenario.results:
            with mlflow.start_run(run_name=f"{name}-seed{r.seed}", nested=True):
                mlflow.log_params({"seed": r.seed, "variant": scenario.variant})
                mlflow.log_metrics(
                    {
                        **r.evaluation.as_dict(),
                        "epochs_run": r.epochs_run,
                        "best_epoch": r.best_epoch,
                        "stopped_early": int(r.stopped_early),
                    }
                )


# --- Grid -----------------------------------------------------------------

def prepare_sample_sets(cfg: Config, acidity, weather, seed: int, with_time_tag=False):
    """Assign splits per station and build every sample set for a variant.

    For blocked splits the eligible years are taken from the Type B sets, which
    are the most restrictive (D-13), so no fold can come out empty.
    """
    arrays = make_weather_arrays(weather)
    stations = list(cfg["data"]["stations"])

    eligible = {}
    if cfg["split"]["method"] == "blocked":
        unsplit = build_sample_sets(acidity, arrays, cfg, types=["B"],
                                    with_time_tag=with_time_tag)
        for st in stations:
            eligible[st] = eligible_years_from([unsplit[(st, "B")]])

    splits = {
        st: assign_splits(acidity, cfg, seed=seed, station=st,
                          eligible_years=eligible.get(st))
        for st in stations
    }
    return build_sample_sets(acidity, arrays, cfg, splits_by_station=splits,
                             with_time_tag=with_time_tag)


def run_variant(
    cfg: Config,
    acidity,
    weather,
    variant: str,
    mlflow=None,
    hidden_sizes=None,
) -> list[ScenarioResult]:
    """Run the full station x type x hidden-size grid for one variant."""
    vcfg = override(cfg, VARIANTS[variant])
    hidden_sizes = list(hidden_sizes or vcfg["model"]["hidden_sizes"])
    seed = int(vcfg["train"]["base_seed"])

    sets = prepare_sample_sets(vcfg, acidity, weather, seed=seed)
    out: list[ScenarioResult] = []

    for (station, wtype), sample_set in sorted(sets.items()):
        counts = realised_counts(sample_set)
        scaler = fit_scaler_for(sample_set, vcfg)

        for hidden in hidden_sizes:
            out.append(
                run_cell(vcfg, sample_set, scaler, hidden,
                         variant=variant, mlflow=mlflow, split_counts=counts)
            )
    return out


def run_cell(
    cfg: Config,
    sample_set,
    scaler: Scaler,
    hidden_size: int,
    kind: str = "lstm",
    variant: str = "paper",
    time_tag: str = "",
    mlflow=None,
    split_counts: dict | None = None,
) -> ScenarioResult:
    """Train one cell (5 repeats) and wrap it as a ScenarioResult."""
    t0 = time.time()
    results = train_repeats(sample_set, scaler, cfg, hidden_size=hidden_size, kind=kind)
    scenario = ScenarioResult(
        variant=variant,
        station=sample_set.station,
        window_type=sample_set.window_type,
        hidden_size=hidden_size,
        n_samples=len(sample_set),
        split_counts=split_counts or realised_counts(sample_set),
        summary=summarise_repeats(results),
        best=best_of(results),
        results=results,
        scaler=scaler,
        seconds=time.time() - t0,
        meta=sample_set.meta,
        kind=kind,
        time_tag=time_tag,
    )
    _log_scenario(mlflow, scenario, cfg)
    log.info(
        "%s %s %s-%s H=%s%s: best MSE %.3f, R %.3f, mean %.3f+-%.3f",
        variant, kind, scenario.station, scenario.window_type, hidden_size,
        f" tag={time_tag}" if time_tag else "",
        scenario.summary["best_all_mse"], scenario.best.evaluation.all.r,
        scenario.summary["all_mse_mean"], scenario.summary["all_mse_sd"],
    )
    return scenario


def run_parametric_study(
    cfg: Config,
    variants=("paper", "blocked", "common", "fit_train"),
    acidity=None,
    weather=None,
    log_mlflow: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Run every requested variant. Returns a tidy table and the raw scenarios."""
    if weather is None:
        weather, _ = clean_weather(cfg)
    if acidity is None:
        acidity, _ = clean_acidity(cfg)

    mlflow = setup_mlflow(cfg) if log_mlflow else None

    scenarios: dict[str, list[ScenarioResult]] = {}
    rows = []
    for variant in variants:
        log.info("=== variant: %s ===", variant)
        scenarios[variant] = run_variant(cfg, acidity, weather, variant, mlflow=mlflow)
        rows.extend(s.row() for s in scenarios[variant])

    return pd.DataFrame(rows), scenarios


def find(scenarios: list[ScenarioResult], station: str, window_type: str, hidden: int):
    """Look up one cell of a variant's grid."""
    for s in scenarios:
        if (s.station, s.window_type, s.hidden_size) == (station, window_type, hidden):
            return s
    raise KeyError(f"No scenario for {station}-{window_type} H={hidden}")


def log_study_artifacts(cfg: Config, table: pd.DataFrame, figs: dict,
                        report_path=None, mlflow=None) -> None:
    """Log the study's figures and reports as artifacts on a summary run.

    Figures are produced after the grid finishes, so they go on their own
    summary run rather than on any single scenario (CLAUDE.md rule 7).
    """
    if mlflow is None:
        mlflow = setup_mlflow(cfg)
    if mlflow is None:
        return

    with mlflow.start_run(run_name="phase6-summary"):
        mlflow.log_params(
            {
                "variants": ",".join(sorted(set(table["variant"]))),
                "n_scenarios": len(table),
                **_flat_config_params(cfg),
            }
        )
        paper = table[table["variant"] == "paper"]
        mlflow.log_metrics(
            {
                "max_abs_diff_vs_paper": float(paper["diff"].abs().max()),
                "mean_abs_diff_vs_paper": float(paper["diff"].abs().mean()),
                "cells_within_0.05": int((paper["diff"].abs() <= 0.05).sum()),
                "mean_repeat_sd": float(paper["mse_sd"].mean()),
            }
        )
        for path in figs.values():
            if Path(path).exists():
                mlflow.log_artifact(str(path), artifact_path="figures")
        if report_path and Path(report_path).exists():
            mlflow.log_artifact(str(report_path))
        mlflow.log_text(table.to_csv(index=False), "table1_raw.csv")
        log.info("Logged Phase 6 figures and reports to MLflow.")


# --- Phase 7: fully connected baseline ------------------------------------

# Paper Section 3.4: R for the FC baseline and the LSTM, Type B, 10 neurons.
PAPER_FC_R = {"BD": 0.64, "C7": 0.42}
PAPER_LSTM_R = {"BD": 0.70, "C7": 0.51}


def run_fc_baseline(cfg: Config, acidity, weather, mlflow=None,
                    variant: str = "paper") -> dict:
    """Phase 7: FC baseline vs LSTM on the Type B input.

    Both models share the same samples, splits and scaler, so their MSEs are
    directly comparable (D-18).
    """
    vcfg = override(cfg, VARIANTS[variant])
    seed = int(vcfg["train"]["base_seed"])
    sets = prepare_sample_sets(vcfg, acidity, weather, seed=seed)

    hidden = int(vcfg["model"]["fc_baseline_hidden"])
    out: dict = {}
    for station in vcfg["data"]["stations"]:
        sample_set = sets[(station, "B")]
        scaler = fit_scaler_for(sample_set, vcfg)
        counts = realised_counts(sample_set)

        out[(station, "lstm")] = run_cell(
            vcfg, sample_set, scaler, hidden, kind="lstm",
            variant=variant, mlflow=mlflow, split_counts=counts,
        )
        out[(station, "fc")] = run_cell(
            vcfg, sample_set, scaler, hidden, kind="fc",
            variant=variant, mlflow=mlflow, split_counts=counts,
        )
    return out


# --- Phase 8: refined model with the time tag -----------------------------

# Paper Section 3.5: BD, Type B, H = 10, with and without the day-number tag.
PAPER_REFINED = {"r": 0.86, "mse": 0.26}
PAPER_ORIGINAL = {"r": 0.70, "mse": 0.54}

TIME_TAG_MODES = ("per_step", "constant")


def run_refined(
    cfg: Config,
    acidity,
    weather,
    mlflow=None,
    stations=("BD", "C7"),
    variant: str = "paper",
    hidden: int = 10,
) -> dict:
    """Phase 8: the time-tag model against the plain one.

    Keyed by `(station, tag_mode)` where tag_mode is "none", "per_step" or
    "constant". The tag does not change which samples survive (Phase 2), so
    the output scaler is identical across all three and their MSEs are
    comparable (D-18).
    """
    vcfg = override(cfg, VARIANTS[variant])
    seed = int(vcfg["train"]["base_seed"])
    out: dict = {}

    # Baseline: no time tag.
    plain = prepare_sample_sets(vcfg, acidity, weather, seed=seed)
    for station in stations:
        s = plain[(station, "B")]
        out[(station, "none")] = run_cell(
            vcfg, s, fit_scaler_for(s, vcfg), hidden, kind="lstm",
            variant=variant, mlflow=mlflow,
        )

    # Refined: one run per time-tag mode.
    for mode in TIME_TAG_MODES:
        mcfg = override(vcfg, {"windows": {"time_tag": mode}})
        tagged = prepare_sample_sets(mcfg, acidity, weather, seed=seed,
                                     with_time_tag=True)
        for station in stations:
            s = tagged[(station, "B")]
            out[(station, mode)] = run_cell(
                mcfg, s, fit_scaler_for(s, mcfg), hidden, kind="lstm",
                variant=variant, time_tag=mode, mlflow=mlflow,
            )
    return out


def scenarios_to_table(scenarios) -> pd.DataFrame:
    """Tidy table from a list or dict of ScenarioResults."""
    values = scenarios.values() if isinstance(scenarios, dict) else scenarios
    return pd.DataFrame([s.row() for s in values])
