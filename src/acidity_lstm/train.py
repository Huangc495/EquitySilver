"""SGDM training loop reproducing MATLAB's `trainNetwork` defaults.

| Setting          | Paper / MATLAB                  | Here                                  |
|------------------|---------------------------------|---------------------------------------|
| Optimizer        | SGDM, momentum 0.9              | `SGD(lr, momentum, weight_decay)`     |
| LR drop          | x0.2 after 125 epochs           | `StepLR(step_size=125, gamma=0.2)`    |
| Mini-batch       | 20                              | 20, final partial batch dropped       |
| Shuffle          | not stated                      | `once` (MATLAB default)               |
| Max epochs       | 200                             | 200                                   |
| Early stopping   | 6 epochs without improvement    | strict improvement on validation MSE  |
| Returned network | not stated                      | the final one (`restore_best: false`) |

Early stopping almost always fires well before epoch 125, so the learning-rate
drop rarely takes effect.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from .config import Config
from .evaluate import Evaluation, evaluate_model
from .models import build_model, set_seed
from .scaling import Scaler
from .splits import SPLIT_NAMES

log = logging.getLogger(__name__)


@dataclass
class TrainResult:
    """One training run: the model, its history and its metrics."""

    model: nn.Module
    evaluation: Evaluation
    scaler: Scaler
    seed: int
    epochs_run: int
    stopped_early: bool
    best_epoch: int
    best_val_mse: float
    history: dict = field(default_factory=dict)
    n_parameters: int = 0

    @property
    def selection_mse(self) -> float:
        """MSE over *all* samples -- how Table 1 selects the best of 5."""
        return self.evaluation.all.mse


def _iter_batches(n: int, batch_size: int, order: np.ndarray, drop_last: bool):
    """Yield index batches from a given ordering."""
    limit = (n // batch_size) * batch_size if drop_last else n
    if drop_last and limit == 0:
        # Fewer samples than one batch: fall back to a single short batch
        # rather than training on nothing.
        limit = n
    for start in range(0, limit, batch_size):
        yield order[start:start + batch_size]


def train_model(
    sample_set,
    scaler: Scaler,
    cfg: Config,
    seed: int,
    hidden_size: int | None = None,
    kind: str = "lstm",
    model: nn.Module | None = None,
) -> TrainResult:
    """Train one model on one SampleSet and evaluate it on every split."""
    tcfg = cfg["train"]
    set_seed(seed)

    n_features = sample_set.X.shape[2]
    if model is None:
        model = build_model(cfg, n_features, hidden_size or 0, kind=kind)

    X = scaler.transform_x(sample_set.X)
    y = scaler.transform_y(sample_set.y)
    split = sample_set.meta["split"].to_numpy()

    idx = {name: np.flatnonzero(split == name) for name in SPLIT_NAMES}
    if len(idx["train"]) == 0:
        raise ValueError("Training split is empty.")

    X_t = torch.as_tensor(X)
    y_t = torch.as_tensor(y)
    train_idx = idx["train"]
    val_idx = idx["val"]

    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(tcfg["lr"]),
        momentum=float(tcfg["momentum"]),
        weight_decay=float(tcfg["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(tcfg["lr_drop_period"]),
        gamma=float(tcfg["lr_drop_factor"]),
    )
    loss_fn = nn.MSELoss()

    batch_size = int(tcfg["batch_size"])
    drop_last = bool(tcfg.get("drop_last", True))
    shuffle = tcfg.get("shuffle", "once")
    max_epochs = int(tcfg["max_epochs"])
    patience = int(tcfg["patience"])
    restore_best = bool(tcfg.get("restore_best", False))

    rng = np.random.default_rng(seed)
    if shuffle == "once":
        order = rng.permutation(train_idx)          # MATLAB shuffles before training only
    elif shuffle == "every_epoch":
        order = train_idx.copy()
    else:
        raise ValueError(f"Unknown train.shuffle {shuffle!r}; expected 'once' or 'every_epoch'.")

    history = {"epoch": [], "train_loss": [], "val_mse": [], "lr": []}
    best_val = float("inf")
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0
    stopped_early = False
    epoch = 0

    for epoch in range(1, max_epochs + 1):
        if shuffle == "every_epoch":
            order = rng.permutation(train_idx)

        model.train()
        epoch_loss, n_batches = 0.0, 0
        for batch in _iter_batches(len(order), batch_size, order, drop_last):
            optimizer.zero_grad()
            loss = loss_fn(model(X_t[batch]), y_t[batch])
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            n_batches += 1
        scheduler.step()

        train_loss = epoch_loss / max(n_batches, 1)
        val_mse = _subset_mse(model, X_t, y_t, val_idx)

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_mse"].append(val_mse)
        history["lr"].append(float(optimizer.param_groups[0]["lr"]))

        # "Stops improving" = no strictly lower validation MSE.
        if np.isfinite(val_mse) and val_mse < best_val:
            best_val = val_mse
            best_epoch = epoch
            epochs_without_improvement = 0
            if restore_best:
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                stopped_early = True
                log.debug("Early stop at epoch %d (best %d, val MSE %.4f).",
                          epoch, best_epoch, best_val)
                break

    if restore_best and best_state is not None:
        model.load_state_dict(best_state)

    return TrainResult(
        model=model,
        evaluation=evaluate_model(model, sample_set, scaler, y_norm=y),
        scaler=scaler,
        seed=seed,
        epochs_run=epoch,
        stopped_early=stopped_early,
        best_epoch=best_epoch,
        best_val_mse=float(best_val),
        history=history,
        n_parameters=sum(p.numel() for p in model.parameters()),
    )


@torch.no_grad()
def _subset_mse(model: nn.Module, X: torch.Tensor, y: torch.Tensor, idx: np.ndarray) -> float:
    """Validation MSE; NaN when the split is empty (no early-stopping signal)."""
    if len(idx) == 0:
        return float("nan")
    model.eval()
    pred = model(X[idx])
    return float(torch.mean((pred - y[idx]) ** 2).item())


def train_repeats(
    sample_set,
    scaler: Scaler,
    cfg: Config,
    hidden_size: int | None = None,
    kind: str = "lstm",
    base_seed: int | None = None,
    n_repeats: int | None = None,
    rebuild_fn=None,
) -> list[TrainResult]:
    """Re-initialise and train `n_repeats` times.

    By default the split is fixed within a scenario, so only the
    initialisation varies across repeats -- the paper's stated reason for
    repeating (`split.resplit_each_repeat: false`).

    When `resplit_each_repeat` is true, `rebuild_fn(seed)` must be supplied; it
    returns a freshly split `(sample_set, scaler)` for that repeat. Re-splitting
    needs the raw measurements, so only the experiment layer can provide it.
    """
    tcfg = cfg["train"]
    base_seed = int(tcfg["base_seed"] if base_seed is None else base_seed)
    n_repeats = int(tcfg["n_repeats"] if n_repeats is None else n_repeats)
    resplit = bool(cfg["split"].get("resplit_each_repeat", False))

    if resplit and rebuild_fn is None:
        raise ValueError(
            "split.resplit_each_repeat is true but no rebuild_fn was supplied; "
            "re-splitting requires the raw measurements, so the caller must "
            "provide a rebuild function."
        )

    results = []
    for r in range(n_repeats):
        seed = base_seed + r
        this_set, this_scaler = (
            rebuild_fn(seed) if resplit else (sample_set, scaler)
        )
        res = train_model(
            this_set, this_scaler, cfg, seed=seed,
            hidden_size=hidden_size, kind=kind,
        )
        log.info(
            "repeat %d/%d seed=%d: all-MSE %.4f, val-MSE %.4f, %d epochs%s",
            r + 1, n_repeats, res.seed, res.selection_mse,
            res.evaluation.by_split["val"].mse, res.epochs_run,
            " (early stop)" if res.stopped_early else "",
        )
        results.append(res)
    return results


def best_of(results: list[TrainResult]) -> TrainResult:
    """Pick the run with the lowest MSE over *all* samples (Table 1's rule)."""
    return min(results, key=lambda r: r.selection_mse)


def summarise_repeats(results: list[TrainResult]) -> dict:
    """Mean, SD and best across repeats, for the Table 1 report."""
    out: dict = {"n_repeats": len(results)}
    for split in ("all", "train", "val", "test"):
        for metric in ("mse", "r", "rmse_mgL"):
            vals = np.array([getattr(r.evaluation.by_split[split], metric) for r in results])
            out[f"{split}_{metric}_mean"] = float(np.nanmean(vals))
            out[f"{split}_{metric}_sd"] = float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0

    best = best_of(results)
    out["best_seed"] = best.seed
    out["best_all_mse"] = best.selection_mse
    for split in ("all", "train", "val", "test"):
        out[f"best_{split}_mse"] = best.evaluation.by_split[split].mse
        out[f"best_{split}_r"] = best.evaluation.by_split[split].r
        out[f"best_{split}_rmse_mgL"] = best.evaluation.by_split[split].rmse_mgL
    out["epochs_mean"] = float(np.mean([r.epochs_run for r in results]))
    out["n_early_stopped"] = int(sum(r.stopped_early for r in results))
    return out
