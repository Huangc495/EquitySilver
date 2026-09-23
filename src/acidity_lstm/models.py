"""The LSTM regressor and the fully connected baseline (paper Section 2).

MATLAB's `lstmLayer` + `fullyConnectedLayer` + `regressionLayer` stack is
reproduced as closely as PyTorch allows; the remaining differences are recorded
in DEVIATIONS.md (D-10, D-11).
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from torch import nn

from .config import Config

log = logging.getLogger(__name__)

# PyTorch packs LSTM gates as (input, forget, cell, output).
GATE_ORDER = ("i", "f", "g", "o")
FORGET_GATE = 1


class LSTMRegressor(nn.Module):
    """One LSTM layer, then a linear map from the final hidden state.

    The hidden state at the last time step is taken (the paper's Fig. 5 reads
    the sequence left to right and predicts at the final step), then a
    `Linear(hidden_size, 1)` produces the normalised acidity.

    `output_activation` defaults to linear. Paper Eq. 7 prints a sigmoid, but a
    sigmoid cannot emit z-scores outside (0, 1); see D-10.
    """

    def __init__(
        self,
        n_features: int,
        hidden_size: int,
        output_activation: str = "linear",
        init: str = "matlab",
        freeze_bias_hh: bool = False,
    ):
        super().__init__()
        self.n_features = int(n_features)
        self.hidden_size = int(hidden_size)
        self.output_activation = output_activation
        self.freeze_bias_hh = bool(freeze_bias_hh)

        self.lstm = nn.LSTM(
            input_size=self.n_features,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.fc = nn.Linear(self.hidden_size, 1)

        if output_activation == "linear":
            self.activation = nn.Identity()
        elif output_activation == "sigmoid":
            self.activation = nn.Sigmoid()
        else:
            raise ValueError(
                f"Unknown output_activation {output_activation!r}; "
                "expected 'linear' or 'sigmoid'."
            )

        if init == "matlab":
            matlab_init(self)
        elif init != "torch_default":
            raise ValueError(f"Unknown init {init!r}; expected 'matlab' or 'torch_default'.")

        if self.freeze_bias_hh:
            freeze_hh_bias(self.lstm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(N, n_steps, n_features) -> (N,)"""
        out, _ = self.lstm(x)
        last = out[:, -1, :]                      # hidden state at the final step
        return self.activation(self.fc(last)).squeeze(-1)


class FCBaseline(nn.Module):
    """Fully connected baseline after Ma et al. (2020a).

    Flattens the whole window (10 steps x n_features) and maps it through one
    hidden layer. The paper does not state the activation; tanh is MATLAB's
    default for this kind of network and is recorded as an assumption (D-09).
    """

    def __init__(
        self,
        n_features: int,
        n_steps: int = 10,
        hidden_size: int = 10,
        activation: str = "tanh",
        init: str = "matlab",
    ):
        super().__init__()
        self.n_features = int(n_features)
        self.n_steps = int(n_steps)
        self.hidden_size = int(hidden_size)

        activations = {"tanh": nn.Tanh, "relu": nn.ReLU, "sigmoid": nn.Sigmoid}
        if activation not in activations:
            raise ValueError(
                f"Unknown activation {activation!r}; expected one of {sorted(activations)}."
            )

        self.flatten = nn.Flatten()
        self.hidden = nn.Linear(self.n_steps * self.n_features, self.hidden_size)
        self.activation = activations[activation]()
        self.out = nn.Linear(self.hidden_size, 1)

        if init == "matlab":
            for layer in (self.hidden, self.out):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        elif init != "torch_default":
            raise ValueError(f"Unknown init {init!r}; expected 'matlab' or 'torch_default'.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(N, n_steps, n_features) -> (N,)"""
        h = self.activation(self.hidden(self.flatten(x)))
        return self.out(h).squeeze(-1)


def matlab_init(model: LSTMRegressor) -> None:
    """Initialise like MATLAB's `lstmLayer` and `fullyConnectedLayer`.

    - `weight_ih`: Glorot (Xavier) uniform
    - `weight_hh`: orthogonal
    - biases: forget gate 1, all others 0, held entirely in `bias_ih`
    - `Linear`: Glorot weights, zero bias
    """
    lstm = model.lstm
    hidden = lstm.hidden_size

    for name, param in lstm.named_parameters():
        if name.startswith("weight_ih"):
            nn.init.xavier_uniform_(param)
        elif name.startswith("weight_hh"):
            # Initialise each gate's square block orthogonally.
            for g in range(len(GATE_ORDER)):
                nn.init.orthogonal_(param.data[g * hidden:(g + 1) * hidden])
        elif name.startswith("bias"):
            nn.init.zeros_(param)

    # MATLAB's default unit forget-gate bias, placed in bias_ih so the two
    # PyTorch bias vectors do not sum to 2.
    with torch.no_grad():
        lstm.bias_ih_l0[FORGET_GATE * hidden:(FORGET_GATE + 1) * hidden].fill_(1.0)

    nn.init.xavier_uniform_(model.fc.weight)
    nn.init.zeros_(model.fc.bias)


def freeze_hh_bias(lstm: nn.LSTM) -> None:
    """Pin `bias_hh` at zero so the effective parameter count matches MATLAB.

    PyTorch keeps a second bias vector that MATLAB does not have (D-11).
    """
    for name, param in lstm.named_parameters():
        if name.startswith("bias_hh"):
            with torch.no_grad():
                param.zero_()
            param.requires_grad_(False)


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    return sum(
        p.numel() for p in model.parameters() if p.requires_grad or not trainable_only
    )


def matlab_parameter_count(n_features: int, hidden_size: int) -> int:
    """Parameters MATLAB would report for the same architecture.

    LSTM: 4H(F + H) weights + 4H biases; FC: H + 1.
    For H = 10, F = 2 this gives 520 + 11 = 531.
    """
    h, f = int(hidden_size), int(n_features)
    return 4 * h * (f + h) + 4 * h + (h + 1)


def build_model(cfg: Config, n_features: int, hidden_size: int, kind: str = "lstm") -> nn.Module:
    """Construct the configured model."""
    mcfg = cfg["model"]
    if kind == "lstm":
        return LSTMRegressor(
            n_features=n_features,
            hidden_size=hidden_size,
            output_activation=mcfg["output_activation"],
            init=mcfg["init"],
            freeze_bias_hh=bool(mcfg.get("freeze_bias_hh", False)),
        )
    if kind == "fc":
        return FCBaseline(
            n_features=n_features,
            n_steps=int(cfg["windows"]["n_steps"]),
            hidden_size=int(mcfg["fc_baseline_hidden"]),
            activation=mcfg["fc_baseline_activation"],
            init=mcfg["init"],
        )
    raise ValueError(f"Unknown model kind {kind!r}; expected 'lstm' or 'fc'.")


@torch.no_grad()
def predict(model: nn.Module, X: np.ndarray, batch_size: int = 512) -> np.ndarray:
    """Normalised predictions for `X`, evaluated in inference mode.

    Lives here rather than in `evaluate` so that serving needs only torch,
    not the plotting stack `evaluate` loads.
    """
    model.eval()
    if len(X) == 0:
        return np.empty(0, dtype=float)

    out = []
    tensor = torch.as_tensor(np.asarray(X, dtype=np.float32))
    for i in range(0, len(tensor), batch_size):
        out.append(model(tensor[i:i + batch_size]).cpu().numpy())
    return np.concatenate(out).astype(float)


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and torch (CLAUDE.md rule 3)."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
