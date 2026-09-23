"""Metrics: MSE in normalised units, Pearson R (Eq. 12), RMSE in mg/L."""

from __future__ import annotations

from dataclasses import dataclass, field

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch import nn

from .models import predict  # noqa: F401  (re-exported: evaluate.predict is public)
from .scaling import Scaler
from .splits import SPLIT_NAMES


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean squared error, in whatever units the arguments are in."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) == 0:
        return float("nan")
    return float(np.mean((y_true - y_pred) ** 2))


def pearson_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Pearson correlation (paper Eq. 12).

    Returns NaN when either series is constant, which is what a correlation
    means there -- undefined, not zero.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) < 2:
        return float("nan")

    a = y_true - y_true.mean()
    b = y_pred - y_pred.mean()
    denom = np.sqrt((a ** 2).sum() * (b ** 2).sum())
    if denom == 0:
        return float("nan")
    return float((a * b).sum() / denom)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mse(y_true, y_pred)))


@dataclass
class SplitMetrics:
    """Metrics for one subset of the samples."""

    n: int
    mse: float
    r: float
    rmse_mgL: float

    def as_dict(self, prefix: str) -> dict:
        return {
            f"{prefix}_n": self.n,
            f"{prefix}_mse": self.mse,
            f"{prefix}_r": self.r,
            f"{prefix}_rmse_mgL": self.rmse_mgL,
        }


@dataclass
class Evaluation:
    """Metrics for every split plus the all-samples set.

    `all` is what best-of-5 selection uses, matching how Table 1 reports the
    "lowest MSE" run.
    """

    by_split: dict = field(default_factory=dict)
    y_true_norm: np.ndarray = field(default_factory=lambda: np.empty(0))
    y_pred_norm: np.ndarray = field(default_factory=lambda: np.empty(0))
    y_true_mgL: np.ndarray = field(default_factory=lambda: np.empty(0))
    y_pred_mgL: np.ndarray = field(default_factory=lambda: np.empty(0))

    @property
    def all(self) -> SplitMetrics:
        return self.by_split["all"]

    def metric(self, split: str, name: str) -> float:
        return getattr(self.by_split[split], name)

    def as_dict(self) -> dict:
        out: dict = {}
        for split, m in self.by_split.items():
            out.update(m.as_dict(split))
        return out


def evaluate_model(
    model: nn.Module,
    sample_set,
    scaler: Scaler,
    y_norm: np.ndarray | None = None,
) -> Evaluation:
    """Evaluate a trained model on every split of one SampleSet.

    MSE and R are computed on normalised values, as the paper reports them;
    RMSE is reported in mg/L after the inverse transform.
    """
    X_norm = scaler.transform_x(sample_set.X)
    y_true_norm = scaler.transform_y(sample_set.y) if y_norm is None else np.asarray(y_norm)
    y_pred_norm = predict(model, X_norm)

    y_true_mgL = scaler.inverse_transform_y(y_true_norm)
    y_pred_mgL = scaler.inverse_transform_y(y_pred_norm)

    split = sample_set.meta["split"].to_numpy()
    ev = Evaluation(
        y_true_norm=np.asarray(y_true_norm, dtype=float),
        y_pred_norm=y_pred_norm,
        y_true_mgL=y_true_mgL,
        y_pred_mgL=y_pred_mgL,
    )

    masks = {"all": np.ones(len(sample_set), dtype=bool)}
    for name in SPLIT_NAMES:
        masks[name] = split == name

    for name, mask in masks.items():
        ev.by_split[name] = SplitMetrics(
            n=int(mask.sum()),
            mse=mse(ev.y_true_norm[mask], ev.y_pred_norm[mask]),
            r=pearson_r(ev.y_true_norm[mask], ev.y_pred_norm[mask]),
            rmse_mgL=rmse(y_true_mgL[mask], y_pred_mgL[mask]),
        )
    return ev


# --- Figures --------------------------------------------------------------

def fit_line(x: np.ndarray, y: np.ndarray):
    """Least-squares fit y = a*x + b, as drawn on the paper's scatter plots."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2 or np.allclose(x, x[0]):
        return float("nan"), float("nan")
    a, b = np.polyfit(x, y, 1)
    return float(a), float(b)


def scatter_measured_vs_calculated(
    evaluations: dict,
    path,
    title: str = "",
    subset: str = "all",
    masks: dict | None = None,
):
    """Paper Fig. 6: normalised measured vs calculated acidity, with fit and R.

    `evaluations` maps a panel label (e.g. "BD") to an `Evaluation`.
    """
    n = len(evaluations)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 5.0), squeeze=False)

    for ax, (label, ev) in zip(axes[0], evaluations.items()):
        mask = (
            np.ones(len(ev.y_true_norm), dtype=bool)
            if masks is None else masks[label]
        )
        x = ev.y_true_norm[mask]
        y = ev.y_pred_norm[mask]
        r = pearson_r(x, y)

        ax.scatter(x, y, s=18, alpha=0.65, edgecolor="none", color="tab:blue")

        lo = float(min(x.min(), y.min()))
        hi = float(max(x.max(), y.max()))
        pad = 0.08 * (hi - lo)
        line = np.array([lo - pad, hi + pad])
        ax.plot(line, line, "k--", lw=1, alpha=0.5, label="1:1")

        a, b = fit_line(x, y)
        if np.isfinite(a):
            ax.plot(line, a * line + b, "r-", lw=1.6, label=f"fit: {a:.2f}x + {b:.2f}")

        ax.set_xlim(line)
        ax.set_ylim(line)
        ax.set_aspect("equal")
        ax.set_xlabel("Measured acidity (normalised)")
        ax.set_ylabel("Calculated acidity (normalised)")
        ax.set_title(f"{label}   R = {r:.2f}   (n = {int(mask.sum())})", fontsize=11)
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left", fontsize=8, framealpha=0.9)

    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def timeseries_measured_vs_calculated(
    meta,
    ev: Evaluation,
    path,
    title: str = "",
    in_mgL: bool = False,
    mark_splits: bool = True,
):
    """Paper Fig. 7: measured and calculated acidity against time."""
    y_true = ev.y_true_mgL if in_mgL else ev.y_true_norm
    y_pred = ev.y_pred_mgL if in_mgL else ev.y_pred_norm
    unit = "mg/L CaCO$_3$" if in_mgL else "normalised"

    dates = pd.to_datetime(meta["date"]) if "date" in meta else np.arange(len(y_true))
    order = np.argsort(np.asarray(dates))
    dates = np.asarray(dates)[order]

    fig, ax = plt.subplots(figsize=(12, 4.2))
    ax.plot(dates, np.asarray(y_true)[order], "o-", ms=3.5, lw=0.8,
            color="tab:blue", label="Measured")
    ax.plot(dates, np.asarray(y_pred)[order], "s-", ms=3.5, lw=0.8,
            color="tab:red", alpha=0.85, label="Calculated")

    if mark_splits and "split" in meta:
        split = meta["split"].to_numpy()[order]
        for name, colour in (("val", "gold"), ("test", "tab:green")):
            idx = np.flatnonzero(split == name)
            if len(idx):
                ax.scatter(dates[idx], np.asarray(y_true)[order][idx],
                           s=52, facecolors="none", edgecolors=colour,
                           linewidths=1.4, label=f"{name} samples", zorder=5)

    ax.set_xlabel("Date")
    ax.set_ylabel(f"Acidity ({unit})")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=9, ncol=2)
    if title:
        ax.set_title(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path

