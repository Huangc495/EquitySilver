"""The registered model's contract: raw daily weather in, mg/L out (MLOPS.md M2).

Synthetic weather and an untrained network are enough here: what matters is
that serving builds exactly the inputs training built, and handles the
input it will meet in production (gaps, disorder, duplicates) without
imputing anything.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from acidity_lstm.config import load_config
from acidity_lstm.models import predict
from acidity_lstm.scaling import fit_scaler
from acidity_lstm.serving import (
    INPUT_COLUMNS,
    OUTPUT_COLUMNS,
    ModelSpec,
    build_network,
    predict_from_weather,
    scaler_from_dict,
    scaler_to_dict,
)
from acidity_lstm.windows import build_samples, make_weather_arrays, window_features


def synthetic_weather(days: int = 400, start: str = "1997-06-01", seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.arange(days)
    return pd.DataFrame({
        "date": pd.date_range(start, periods=days, freq="D"),
        "precip_mm": rng.gamma(0.8, 3.0, days),
        "tmean_c": 5 + 10 * np.sin(2 * np.pi * t / 365) + rng.normal(0, 2, days),
    })


def sample_dates(weather: pd.DataFrame, every: int = 13) -> pd.DataFrame:
    dates = weather["date"].iloc[::every].reset_index(drop=True)
    return pd.DataFrame({"station": "BD", "date": dates,
                         "acidity_mgL": np.linspace(5000, 30000, len(dates))})


@pytest.fixture
def cfg():
    return load_config()


def make_spec(cfg, window_type="B", with_tag=False) -> ModelSpec:
    names = ("precip", "tmean") + (("day_number",) if with_tag else ())
    return ModelSpec(
        station="BD", role="challenger" if with_tag else "champion",
        window_type=window_type, n_steps=int(cfg["windows"]["n_steps"]),
        hidden_size=10, n_features=len(names), feature_names=names,
        output_activation="linear", freeze_bias_hh=False,
        with_time_tag=with_tag, time_tag_mode="per_step",
        time_tag_origin=str(cfg["windows"]["time_tag_origin"]),
    )


def trained_like(cfg, weather, window_type="B", with_tag=False):
    """A SampleSet, a scaler fitted on it and a seeded, untrained network."""
    s = build_samples(sample_dates(weather), weather, cfg, "BD", window_type,
                      with_time_tag=with_tag)
    scaler = fit_scaler(s.X, s.y, s.feature_names)
    spec = make_spec(cfg, window_type, with_tag)
    torch.manual_seed(0)
    return s, scaler, spec, build_network(spec)


# --- one definition of a window -------------------------------------------

@pytest.mark.parametrize("window_type", ["A", "B"])
@pytest.mark.parametrize("with_tag", [False, True])
def test_window_features_are_the_training_inputs(cfg, window_type, with_tag):
    weather = synthetic_weather()
    weather.loc[150, "tmean_c"] = np.nan                    # knock out some windows
    acidity = sample_dates(weather)
    s = build_samples(acidity, weather, cfg, "BD", window_type, with_time_tag=with_tag)

    feat, complete, _ = window_features(
        make_weather_arrays(weather), acidity["date"], window_type,
        int(cfg["windows"]["n_steps"]), with_time_tag=with_tag,
        time_tag_mode=cfg["windows"]["time_tag"],
        time_tag_origin=cfg["windows"]["time_tag_origin"],
    )
    assert complete.sum() == len(s)
    np.testing.assert_array_equal(feat[complete].astype(np.float32), s.X)


# --- serving reproduces the training path ---------------------------------

@pytest.mark.parametrize("with_tag", [False, True])
def test_serving_reproduces_the_training_prediction(cfg, with_tag):
    """Raw weather through serving == SampleSet through the training code."""
    weather = synthetic_weather()
    s, scaler, spec, model = trained_like(cfg, weather, with_tag=with_tag)

    expected = scaler.inverse_transform_y(predict(model, scaler.transform_x(s.X)))
    out = predict_from_weather(weather, spec, scaler, model).set_index("date")
    got = out.loc[pd.DatetimeIndex(s.meta["date"]), "acidity_mgL"].to_numpy()
    np.testing.assert_array_equal(got, expected)


def test_output_has_one_row_per_input_row_in_input_order(cfg):
    weather = synthetic_weather()
    _, scaler, spec, model = trained_like(cfg, weather)
    ordered = predict_from_weather(weather, spec, scaler, model)

    shuffled_in = weather.sample(frac=1.0, random_state=3).reset_index(drop=True)
    shuffled = predict_from_weather(shuffled_in, spec, scaler, model)

    assert list(shuffled.columns) == list(OUTPUT_COLUMNS)
    assert len(shuffled) == len(weather)
    assert (shuffled["date"].to_numpy() == shuffled_in["date"].to_numpy()).all()
    a = ordered.set_index("date")["acidity_mgL"]
    b = shuffled.set_index("date")["acidity_mgL"].reindex(a.index)
    np.testing.assert_array_equal(a.to_numpy(), b.to_numpy())


def test_a_type_b_prediction_needs_seventy_days(cfg):
    weather = synthetic_weather(days=100)
    _, scaler, spec, model = trained_like(cfg, synthetic_weather())
    out = predict_from_weather(weather, spec, scaler, model)
    assert not out["window_complete"].iloc[:69].any()
    assert out["window_complete"].iloc[69:].all()
    assert out["acidity_mgL"].iloc[:69].isna().all()
    assert out["acidity_mgL"].iloc[69:].notna().all()


def test_a_missing_day_blanks_every_window_covering_it(cfg):
    weather = synthetic_weather(days=250)
    weather.loc[150, "precip_mm"] = np.nan
    _, scaler, spec, model = trained_like(cfg, synthetic_weather())
    out = predict_from_weather(weather, spec, scaler, model)
    covered = np.arange(150, 150 + 70)
    assert not out["window_complete"].iloc[covered].any()
    assert out["window_complete"].iloc[220:].all()
    assert out["window_complete"].iloc[69:150].all()


def test_an_absent_row_counts_as_a_missing_day(cfg):
    """Production feeds will skip days; that must not shift the calendar."""
    weather = synthetic_weather(days=250)
    _, scaler, spec, model = trained_like(cfg, synthetic_weather())
    with_nan = weather.copy()
    with_nan.loc[150, ["precip_mm", "tmean_c"]] = np.nan
    dropped = weather.drop(index=150).reset_index(drop=True)

    a = predict_from_weather(with_nan, spec, scaler, model).drop(index=150).reset_index(drop=True)
    b = predict_from_weather(dropped, spec, scaler, model)
    pd.testing.assert_frame_equal(a, b)


def test_conflicting_duplicate_dates_are_refused(cfg):
    weather = synthetic_weather(days=120)
    _, scaler, spec, model = trained_like(cfg, synthetic_weather())
    twin = weather.iloc[[100]].copy()
    predict_from_weather(pd.concat([weather, twin]), spec, scaler, model)   # identical: fine
    twin["tmean_c"] += 1.0
    with pytest.raises(ValueError, match="different values"):
        predict_from_weather(pd.concat([weather, twin]), spec, scaler, model)


def test_missing_columns_are_named(cfg):
    _, scaler, spec, model = trained_like(cfg, synthetic_weather())
    with pytest.raises(ValueError, match="tmean_c"):
        predict_from_weather(synthetic_weather().drop(columns="tmean_c"), spec, scaler, model)


def test_time_tag_is_reported_only_for_a_tagged_model(cfg):
    weather = synthetic_weather()
    _, scaler, spec, model = trained_like(cfg, weather)
    assert predict_from_weather(weather, spec, scaler, model)["time_tag_z"].isna().all()

    _, scaler, spec, model = trained_like(cfg, weather, with_tag=True)
    out = predict_from_weather(weather, spec, scaler, model)
    origin = pd.Timestamp(spec.time_tag_origin)
    day = (out["date"] - origin).dt.days + 1
    i = spec.feature_names.index("day_number")
    np.testing.assert_allclose(out["time_tag_z"], (day - scaler.x_mean[i]) / scaler.x_std[i])


# --- what travels with the model -------------------------------------------

def test_spec_and_scaler_survive_json(cfg):
    weather = synthetic_weather()
    _, scaler, spec, _ = trained_like(cfg, weather, with_tag=True)
    assert ModelSpec.from_dict(json.loads(json.dumps(spec.to_dict()))) == spec
    back = scaler_from_dict(json.loads(json.dumps(scaler_to_dict(scaler))))
    np.testing.assert_array_equal(back.x_mean, scaler.x_mean)
    np.testing.assert_array_equal(back.x_std, scaler.x_std)
    assert (back.y_mean, back.y_std, back.feature_names) == (scaler.y_mean, scaler.y_std,
                                                             scaler.feature_names)


def test_logged_model_round_trips_through_mlflow(cfg, tmp_path):
    """Log, load in pyfunc form, predict: identical to the in-process result."""
    import mlflow

    from acidity_lstm.registry import log_model

    weather = synthetic_weather()
    s, scaler, spec, model = trained_like(cfg, weather, with_tag=True)
    example = weather.iloc[-70:].reset_index(drop=True)

    previous = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(tmp_path.as_uri())
    try:
        mlflow.set_experiment("serving-round-trip")
        with mlflow.start_run():
            uri = log_model(mlflow, [model], scaler, cfg, spec, example,
                            cfg["registry"]["pip_requirements"])
        loaded = mlflow.pyfunc.load_model(uri)
    finally:
        mlflow.set_tracking_uri(previous)

    sig = loaded.metadata.signature
    assert sig.inputs.input_names() == list(INPUT_COLUMNS)
    assert sig.outputs.input_names() == list(OUTPUT_COLUMNS)
    pd.testing.assert_frame_equal(loaded.predict(weather),
                                  predict_from_weather(weather, spec, scaler, model))
