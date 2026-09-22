"""Phase 9 (two-year forecast) and Phase 10 (weather sensitivity).

Kept apart from `experiments.py` because both differ from the parametric study
in ways that matter:

- the scaler is fitted on the training period only, so these MSEs are **not**
  comparable with Table 1's (DEVIATIONS.md D-18);
- best-of-5 is selected on train+validation MSE, never on the forecast period;
- the split is time-based rather than random.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import Config
from .evaluate import Evaluation, predict
from .experiments import _flat_config_params
from .scaling import Scaler, fit_scaler
from .splits import random_split, realised_counts
from .train import TrainResult, train_repeats
from .windows import build_sample_sets, make_weather_arrays

log = logging.getLogger(__name__)

# Paper Section 3.6.
PAPER_FORECAST = {"train_val_mse": 0.28, "predict_mse": 0.23}

# The forecast period reuses the "test" split slot so the existing evaluation
# code reports it without special-casing.
PREDICT_SPLIT = "test"

VARIABLE_COLUMNS = {"precip": "precip_mm", "tmean": "tmean_c"}
VARIABLE_LABELS = {"precip": "Precipitation", "tmean": "Temperature"}


def combined_mse(evaluation: Evaluation, splits) -> float:
    """Pooled MSE over several splits, weighted by sample count."""
    total, n = 0.0, 0
    for name in splits:
        m = evaluation.by_split[name]
        if m.n and np.isfinite(m.mse):
            total += m.mse * m.n
            n += m.n
    return total / n if n else float("nan")


@dataclass
class ForecastResult:
    """A model trained on the training period and scored on the forecast."""

    sample_set: object
    scaler: Scaler
    results: list
    best: TrainResult
    train_years: tuple
    predict_years: tuple
    split_counts: dict
    seconds: float = 0.0

    @property
    def meta(self):
        return self.sample_set.meta

    def train_val_mse(self, result: TrainResult | None = None) -> float:
        """MSE over the training period (train and validation pooled)."""
        return combined_mse((result or self.best).evaluation, ("train", "val"))

    def predict_mse(self, result: TrainResult | None = None) -> float:
        return (result or self.best).evaluation.by_split[PREDICT_SPLIT].mse

    def predict_rmse_mgL(self, result: TrainResult | None = None) -> float:
        return (result or self.best).evaluation.by_split[PREDICT_SPLIT].rmse_mgL

    def spread(self) -> dict:
        """Forecast-period MSE across the repeats."""
        vals = np.array([self.predict_mse(r) for r in self.results])
        return {
            "values": vals,
            "mean": float(np.mean(vals)),
            "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        }


def build_forecast_samples(cfg: Config, acidity, weather, seed: int):
    """BD Type B samples with the time tag, split into training and forecast.

    Splits are assigned on the *surviving* training-period samples so the
    80/20 ratio is exact. The forecast period is time-based, so that choice
    cannot affect it (DEVIATIONS.md D-25).
    """
    fcfg = cfg["forecast"]
    station = fcfg["station"]
    tr0, tr1 = fcfg["train_years"]
    p0, p1 = fcfg["predict_years"]

    full = build_sample_sets(
        acidity, make_weather_arrays(weather), cfg,
        stations=[station], types=["B"], with_time_tag=True,
    )[(station, "B")]

    years = full.meta["date"].dt.year.to_numpy()
    in_train = (years >= tr0) & (years <= tr1)
    in_predict = (years >= p0) & (years <= p1)
    kept = full.select(in_train | in_predict)

    kyears = kept.meta["date"].dt.year.to_numpy()
    k_train = (kyears >= tr0) & (kyears <= tr1)

    labels = np.empty(len(kept), dtype=object)
    labels[~k_train] = PREDICT_SPLIT
    labels[k_train] = random_split(
        int(k_train.sum()), [*fcfg["fractions"], 0.0], seed
    )
    kept = kept.with_splits(labels)

    # The scaler sees the training period only: the honest choice for a
    # forecast, and a deliberate departure from the paper's fit-on-all default.
    scaler = fit_scaler(
        kept.X[k_train], kept.y[k_train], kept.feature_names,
        ddof=int(cfg["scaling"]["ddof"]),
        fit_on=str(fcfg.get("scaler_fit_on", "train_period")),
        station=station, window_type="B",
    )
    return kept, scaler


def run_forecast(cfg: Config, acidity, weather, mlflow=None,
                 hidden: int = 10) -> ForecastResult:
    """Train on the training period and predict the forecast period.

    Best-of-5 is selected on train+validation MSE. Selecting on the forecast
    period would leak the answer we are trying to measure.
    """
    fcfg = cfg["forecast"]
    seed = int(cfg["train"]["base_seed"])
    t0 = time.time()

    sample_set, scaler = build_forecast_samples(cfg, acidity, weather, seed)
    results = train_repeats(sample_set, scaler, cfg, hidden_size=hidden)

    out = ForecastResult(
        sample_set=sample_set,
        scaler=scaler,
        results=results,
        best=min(results,
                 key=lambda r: combined_mse(r.evaluation, ("train", "val"))),
        train_years=tuple(fcfg["train_years"]),
        predict_years=tuple(fcfg["predict_years"]),
        split_counts=realised_counts(sample_set),
        seconds=time.time() - t0,
    )
    _log_forecast(mlflow, out, cfg)
    log.info(
        "forecast: train+val MSE %.3f (paper %.2f), predict MSE %.3f (paper %.2f)",
        out.train_val_mse(), PAPER_FORECAST["train_val_mse"],
        out.predict_mse(), PAPER_FORECAST["predict_mse"],
    )
    return out


def _log_forecast(mlflow, fc: ForecastResult, cfg: Config) -> None:
    if mlflow is None:
        return
    with mlflow.start_run(run_name="phase9-forecast"):
        mlflow.log_params(
            {
                "station": cfg["forecast"]["station"],
                "train_years": str(fc.train_years),
                "predict_years": str(fc.predict_years),
                "scaler_fit_on": fc.scaler.fit_on,
                **_flat_config_params(cfg),
            }
        )
        spread = fc.spread()
        mlflow.log_metrics(
            {
                "train_val_mse": fc.train_val_mse(),
                "predict_mse": fc.predict_mse(),
                "paper_train_val_mse": PAPER_FORECAST["train_val_mse"],
                "paper_predict_mse": PAPER_FORECAST["predict_mse"],
                "predict_mse_mean": spread["mean"],
                "predict_mse_sd": spread["sd"],
                "predict_rmse_mgL": fc.predict_rmse_mgL(),
                **{f"n_{k}": v for k, v in fc.split_counts.items()},
            }
        )
        mlflow.log_dict(
            {
                "feature_names": list(fc.scaler.feature_names),
                "x_mean": [float(v) for v in fc.scaler.x_mean],
                "x_std": [float(v) for v in fc.scaler.x_std],
                "y_mean": float(fc.scaler.y_mean),
                "y_std": float(fc.scaler.y_std),
            },
            "forecast_scaler.json",
        )
        for r in fc.results:
            with mlflow.start_run(run_name=f"phase9-seed{r.seed}", nested=True):
                mlflow.log_params({"seed": r.seed})
                mlflow.log_metrics(
                    {
                        **r.evaluation.as_dict(),
                        "train_val_mse": combined_mse(r.evaluation, ("train", "val")),
                        "epochs_run": r.epochs_run,
                    }
                )


# --- Phase 10: sensitivity ------------------------------------------------

def perturb_weather(weather, variable: str, factor: float, start, end):
    """Multiply one raw daily variable by `factor` over a date range.

    Only days inside [start, end] are scaled, so windows reaching back before
    the period keep their real weather.
    """
    if variable not in VARIABLE_COLUMNS:
        raise ValueError(
            "Unknown variable " + repr(variable) + "; expected one of "
            + str(sorted(VARIABLE_COLUMNS))
        )
    column = VARIABLE_COLUMNS[variable]
    out = weather.copy()
    mask = out["date"].between(pd.Timestamp(start), pd.Timestamp(end))
    out.loc[mask, column] = out.loc[mask, column] * float(factor)
    return out


def run_sensitivity(cfg: Config, acidity, weather, forecast: ForecastResult,
                    mlflow=None):
    """Re-predict the forecast period under perturbed weather.

    The Phase 9 model and its scaler are used unchanged; only the raw weather
    of the prediction period is altered. Returns `(table, curves, meta)`.
    """
    scfg = cfg["sensitivity"]
    fcfg = cfg["forecast"]
    p0, p1 = fcfg["predict_years"]
    start, end = f"{p0}-01-01", f"{p1}-12-31"
    seed = int(cfg["train"]["base_seed"])

    model, scaler = forecast.best.model, forecast.scaler
    base_dates = forecast.meta.loc[
        forecast.meta["split"] == PREDICT_SPLIT, "date"
    ].to_numpy()

    def predict_period(w):
        s, _ = build_forecast_samples(cfg, acidity, w, seed)
        s = s.select((s.meta["split"] == PREDICT_SPLIT).to_numpy())
        # A perturbation must not silently change which samples survive,
        # or the curves would not be comparable point by point.
        if not np.array_equal(s.meta["date"].to_numpy(), base_dates):
            raise ValueError("Perturbation changed the surviving forecast samples.")
        return s, scaler.inverse_transform_y(predict(model, scaler.transform_x(s.X)))

    base_set, base_pred = predict_period(weather)
    base_mean = float(np.mean(base_pred))

    rows = [
        {
            "scenario": "Real weather",
            "variable": "",
            "factor": 1.0,
            "mean_pred_mgL": base_mean,
            "mean_change_mgL": 0.0,
            "mean_change_pct": 0.0,
        }
    ]
    curves = {"Real weather": base_pred}

    for variable in scfg["variables"]:
        for factor in scfg["factors"]:
            w = perturb_weather(weather, variable, factor, start, end)
            _, pred = predict_period(w)
            label = VARIABLE_LABELS[variable] + " x" + format(factor, "g")
            curves[label] = pred
            change = float(np.mean(pred - base_pred))
            rows.append(
                {
                    "scenario": label,
                    "variable": variable,
                    "factor": factor,
                    "mean_pred_mgL": float(np.mean(pred)),
                    "mean_change_mgL": change,
                    "mean_change_pct": 100.0 * change / base_mean,
                }
            )

    table = pd.DataFrame(rows)
    _log_sensitivity(mlflow, table, cfg)
    return table, curves, base_set.meta


def _log_sensitivity(mlflow, table: pd.DataFrame, cfg: Config) -> None:
    if mlflow is None:
        return
    with mlflow.start_run(run_name="phase10-sensitivity"):
        mlflow.log_params(
            {
                "factors": str(cfg["sensitivity"]["factors"]),
                "variables": str(cfg["sensitivity"]["variables"]),
            }
        )
        for _, r in table.iterrows():
            if not r["variable"]:
                continue
            key = r["variable"] + "_x" + str(r["factor"]).replace(".", "_")
            mlflow.log_metrics(
                {
                    key + "_change_mgL": float(r["mean_change_mgL"]),
                    key + "_change_pct": float(r["mean_change_pct"]),
                }
            )
        mlflow.log_text(table.to_csv(index=False), "sensitivity.csv")
