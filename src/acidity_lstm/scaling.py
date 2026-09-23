"""Z-score normalisation (paper Eqs. 8-10).

Eq. 9 is printed without the square, but it is plainly meant to be the standard
deviation. We use the population standard deviation (`ddof = 0`), which is what
MATLAB's `normalize`/`zscore` default would give for a fixed population.

One mean and std per input feature, pooled over every time step of every sample
in the fitting set; one mean and std for the output, per station.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .config import Config

log = logging.getLogger(__name__)

# Guards against dividing by zero if a feature is constant in the fitting set.
_MIN_STD = 1e-12


@dataclass
class Scaler:
    """Fitted z-score parameters for one scenario.

    A scenario is one station x input type x time-tag setting, so each gets its
    own scaler (CLAUDE.md Phase 3, "Scope").
    """

    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: float
    y_std: float
    feature_names: tuple
    ddof: int
    fit_on: str
    n_fit: int
    station: str = ""
    window_type: str = ""

    # --- transforms ------------------------------------------------------

    def transform_x(self, X: np.ndarray) -> np.ndarray:
        """Normalise inputs of shape (N, n_steps, n_features)."""
        if X.shape[-1] != len(self.x_mean):
            raise ValueError(
                f"Expected {len(self.x_mean)} features, got {X.shape[-1]}."
            )
        return ((X - self.x_mean) / self.x_std).astype(np.float32)

    def transform_y(self, y: np.ndarray) -> np.ndarray:
        return ((np.asarray(y, dtype=float) - self.y_mean) / self.y_std).astype(np.float32)

    def inverse_transform_y(self, z: np.ndarray) -> np.ndarray:
        """Back to mg/L, for RMSE in physical units."""
        return (np.asarray(z, dtype=float) * self.y_std + self.y_mean).astype(float)

    def transform(self, X: np.ndarray, y: np.ndarray):
        return self.transform_x(X), self.transform_y(y)

    # --- logging ---------------------------------------------------------

    @property
    def params(self) -> dict:
        """Flat dict of scaler parameters, for MLflow logging."""
        out = {
            "scaler_fit_on": self.fit_on,
            "scaler_ddof": self.ddof,
            "scaler_n_fit": self.n_fit,
            "scaler_y_mean": float(self.y_mean),
            "scaler_y_std": float(self.y_std),
        }
        for name, m, s in zip(self.feature_names, self.x_mean, self.x_std):
            out[f"scaler_x_mean_{name}"] = float(m)
            out[f"scaler_x_std_{name}"] = float(s)
        return out


def fit_scaler(
    X: np.ndarray,
    y: np.ndarray,
    feature_names,
    ddof: int = 0,
    fit_on: str = "all",
    station: str = "",
    window_type: str = "",
) -> Scaler:
    """Fit z-score parameters on the given (already selected) rows.

    Input statistics are pooled over samples *and* time steps, giving one mean
    and one std per feature.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.ndim != 3:
        raise ValueError(f"Expected X of shape (N, n_steps, n_features), got {X.shape}.")
    if len(X) == 0:
        raise ValueError("Cannot fit a scaler on an empty set.")

    x_mean = X.mean(axis=(0, 1))
    x_std = X.std(axis=(0, 1), ddof=ddof)
    y_mean = float(y.mean())
    y_std = float(y.std(ddof=ddof))

    for name, s in zip(feature_names, x_std):
        if s < _MIN_STD:
            log.warning("Feature %r is constant in the fitting set; std pinned to 1.", name)
    x_std = np.where(x_std < _MIN_STD, 1.0, x_std)
    if y_std < _MIN_STD:
        log.warning("Output is constant in the fitting set; std pinned to 1.")
        y_std = 1.0

    return Scaler(
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
        feature_names=tuple(feature_names),
        ddof=int(ddof),
        fit_on=fit_on,
        n_fit=len(X),
        station=station,
        window_type=window_type,
    )


def fit_scaler_for(sample_set, cfg: Config, fit_on: str | None = None) -> Scaler:
    """Fit a scaler for one SampleSet, honouring `scaling.fit_on`.

    `all` (the paper's choice) fits on every sample in the scenario, which
    leaks test-set statistics into training. `train` fits on the training rows
    only. Phase 6 reports both.
    """
    scfg = cfg["scaling"]
    fit_on = fit_on or scfg["fit_on"]
    ddof = int(scfg.get("ddof", 0))

    if fit_on == "all":
        mask = np.ones(len(sample_set), dtype=bool)
    elif fit_on == "train":
        split = sample_set.meta["split"].to_numpy()
        mask = split == "train"
        if not mask.any():
            raise ValueError(
                "scaling.fit_on='train' but no rows are assigned to the training "
                "split; assign splits before fitting the scaler."
            )
    else:
        raise ValueError(f"Unknown scaling.fit_on {fit_on!r}; expected 'all' or 'train'.")

    return fit_scaler(
        sample_set.X[mask],
        sample_set.y[mask],
        sample_set.feature_names,
        ddof=ddof,
        fit_on=fit_on,
        station=sample_set.station,
        window_type=sample_set.window_type,
    )
