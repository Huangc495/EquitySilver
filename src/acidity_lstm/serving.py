"""The registered model: raw daily weather in, acidity in mg/L out (MLOPS.md M2).

Serving reuses `windows.window_features`, the function that built the
training tensors, and the scenario's own fitted scaler. Shipping weights
without the scaler, or re-implementing the windows, is how training/serving
skew creeps in (HANDOVER.md section 4). Nothing here needs the YAML config:
everything a prediction depends on travels with the model as a `ModelSpec`,
the scaler parameters and the weights.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import mlflow.pyfunc
import numpy as np
import pandas as pd
import torch

from .models import LSTMRegressor, predict
from .scaling import Scaler
from .windows import make_weather_arrays, window_features

INPUT_COLUMNS = ("date", "precip_mm", "tmean_c")
OUTPUT_COLUMNS = ("date", "acidity_mgL", "window_complete", "time_tag_z")


@dataclass(frozen=True)
class ModelSpec:
    """Everything besides weights and scaler that a prediction depends on."""

    station: str
    role: str                   # "champion" | "challenger"
    window_type: str
    n_steps: int
    hidden_size: int
    n_features: int
    feature_names: tuple
    output_activation: str
    freeze_bias_hh: bool
    with_time_tag: bool
    time_tag_mode: str
    time_tag_origin: str
    n_members: int = 1          # networks averaged (MLOPS.md D10)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["feature_names"] = list(self.feature_names)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ModelSpec":
        return cls(**{**d, "feature_names": tuple(d["feature_names"])})


def scaler_to_dict(scaler: Scaler) -> dict:
    return {
        "x_mean": [float(v) for v in scaler.x_mean],
        "x_std": [float(v) for v in scaler.x_std],
        "y_mean": float(scaler.y_mean),
        "y_std": float(scaler.y_std),
        "feature_names": list(scaler.feature_names),
        "ddof": int(scaler.ddof),
        "fit_on": scaler.fit_on,
        "n_fit": int(scaler.n_fit),
        "station": scaler.station,
        "window_type": scaler.window_type,
    }


def scaler_from_dict(d: dict) -> Scaler:
    return Scaler(
        x_mean=np.asarray(d["x_mean"], dtype=float),
        x_std=np.asarray(d["x_std"], dtype=float),
        y_mean=float(d["y_mean"]),
        y_std=float(d["y_std"]),
        feature_names=tuple(d["feature_names"]),
        ddof=int(d["ddof"]),
        fit_on=d["fit_on"],
        n_fit=int(d["n_fit"]),
        station=d.get("station", ""),
        window_type=d.get("window_type", ""),
    )


def build_network(spec: ModelSpec) -> LSTMRegressor:
    """An untrained network of the spec's shape, ready for `load_state_dict`."""
    return LSTMRegressor(
        n_features=spec.n_features,
        hidden_size=spec.hidden_size,
        output_activation=spec.output_activation,
        init="torch_default",           # overwritten by the saved weights
        freeze_bias_hh=spec.freeze_bias_hh,
    )


def _daily_calendar(weather: pd.DataFrame) -> pd.DataFrame:
    """The input's days on a continuous calendar; absent days become NaN.

    Repeated dates are accepted only when they agree, since picking one of
    two different readings would be silent imputation.
    """
    daily = weather.loc[:, list(INPUT_COLUMNS)]
    dup = daily["date"].duplicated(keep=False)
    if dup.any():
        conflicting = daily.loc[dup].groupby("date")[["precip_mm", "tmean_c"]].nunique(dropna=False)
        bad = conflicting[(conflicting > 1).any(axis=1)]
        if len(bad):
            raise ValueError(
                f"{len(bad)} date(s) appear more than once with different values, "
                f"first {bad.index[0].date()}."
            )
    daily = daily.drop_duplicates("date").set_index("date").sort_index()
    calendar = pd.date_range(daily.index.min(), daily.index.max(), freq="D", name="date")
    return daily.reindex(calendar).reset_index()


def predict_members(models, X: np.ndarray) -> np.ndarray:
    """Normalised prediction averaged over the ensemble's networks (D10).

    One network is an ensemble of one, and returns exactly `predict`.
    """
    members = models if isinstance(models, (list, tuple)) else [models]
    if len(members) == 1:
        return predict(members[0], X)
    return np.mean(np.stack([predict(m, X) for m in members]), axis=0)


def predict_from_weather(
    weather: pd.DataFrame,
    spec: ModelSpec,
    scaler: Scaler,
    model,
) -> pd.DataFrame:
    """Predicted acidity for a sample taken on each input day.

    `model` is one network or a list of them; a list is averaged in
    normalised units before the inverse transform (D10).

    One output row per input row, in input order. A day's prediction needs
    the complete window ending on it (70 days for Type B) inside the input;
    otherwise `acidity_mgL` is NaN and `window_complete` False. Nothing is
    imputed. `time_tag_z` is the normalised day-number feature at the
    window's last step (challenger only; NaN for a model without the tag),
    so monitoring can see how far past its training range the tag has gone.
    """
    missing = [c for c in INPUT_COLUMNS if c not in weather.columns]
    if missing:
        raise ValueError(f"Weather input is missing column(s) {missing}.")
    if len(weather) == 0:
        return pd.DataFrame({c: pd.Series(dtype=t) for c, t in zip(
            OUTPUT_COLUMNS, ("datetime64[ns]", float, bool, float))})

    rows = weather.loc[:, list(INPUT_COLUMNS)].copy()
    rows["date"] = pd.to_datetime(rows["date"]).dt.normalize()
    if rows["date"].isna().any():
        raise ValueError("Weather input has rows without a date.")
    for col in ("precip_mm", "tmean_c"):
        rows[col] = pd.to_numeric(rows[col], errors="coerce").astype(float)

    arrays = make_weather_arrays(_daily_calendar(rows))
    feat, complete, day_number = window_features(
        arrays, rows["date"], spec.window_type, spec.n_steps,
        with_time_tag=spec.with_time_tag,
        time_tag_mode=spec.time_tag_mode,
        time_tag_origin=spec.time_tag_origin,
    )

    acidity = np.full(len(rows), np.nan)
    if complete.any():
        # float32 first, exactly as SampleSet stores training inputs.
        X = scaler.transform_x(feat[complete].astype(np.float32))
        acidity[complete] = scaler.inverse_transform_y(predict_members(model, X))

    time_tag_z = np.full(len(rows), np.nan)
    if spec.with_time_tag:
        i = spec.feature_names.index("day_number")
        time_tag_z = (day_number - scaler.x_mean[i]) / scaler.x_std[i]

    return pd.DataFrame({
        "date": rows["date"].to_numpy(),
        "acidity_mgL": acidity,
        "window_complete": complete.astype(bool),
        "time_tag_z": np.asarray(time_tag_z, dtype=float),
    })


class AcidityModel(mlflow.pyfunc.PythonModel):
    """MLflow pyfunc around `predict_from_weather`.

    Artifacts: `spec` (ModelSpec JSON), `scaler` (scaler JSON) and `weights`
    (a list of torch state_dicts, one per ensemble member). The package itself
    travels as `code_paths`, so the loaded model runs the same window code
    that trained it.
    """

    def load_context(self, context) -> None:
        with open(context.artifacts["spec"], encoding="utf-8") as fh:
            self.spec = ModelSpec.from_dict(json.load(fh))
        with open(context.artifacts["scaler"], encoding="utf-8") as fh:
            self.scaler = scaler_from_dict(json.load(fh))
        states = torch.load(context.artifacts["weights"], map_location="cpu", weights_only=True)
        if isinstance(states, dict):            # a single state_dict: one member
            states = [states]
        if len(states) != self.spec.n_members:
            raise ValueError(f"Spec says {self.spec.n_members} member(s); "
                             f"weights hold {len(states)}.")
        self.model = []
        for state in states:
            net = build_network(self.spec)
            net.load_state_dict(state)
            net.eval()
            self.model.append(net)

    def predict(self, context, model_input: pd.DataFrame, params=None) -> pd.DataFrame:
        return predict_from_weather(model_input, self.spec, self.scaler, self.model)


def signature():
    """Input and output schema; Unity Catalog requires both."""
    from mlflow.models import ModelSignature
    from mlflow.types.schema import ColSpec, Schema

    return ModelSignature(
        inputs=Schema([
            ColSpec("datetime", "date"),
            ColSpec("double", "precip_mm"),
            ColSpec("double", "tmean_c"),
        ]),
        outputs=Schema([
            ColSpec("datetime", "date"),
            ColSpec("double", "acidity_mgL"),
            ColSpec("boolean", "window_complete"),
            ColSpec("double", "time_tag_z"),
        ]),
    )
