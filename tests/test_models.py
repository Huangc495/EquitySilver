"""Model tests required by CLAUDE.md Phase 4."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from acidity_lstm.config import Config, load_config
from acidity_lstm.models import (
    FORGET_GATE,
    FCBaseline,
    LSTMRegressor,
    build_model,
    count_parameters,
    matlab_parameter_count,
    set_seed,
)

N_STEPS = 10


@pytest.fixture
def cfg():
    return load_config()


def with_model(cfg: Config, **updates) -> Config:
    import copy

    raw = copy.deepcopy(cfg.raw)
    raw["model"].update(updates)
    return Config(raw=raw, paths=cfg.paths, repo_root=cfg.repo_root)


# --- Output shapes --------------------------------------------------------

@pytest.mark.parametrize("hidden", [5, 10, 20])
@pytest.mark.parametrize("n_features", [2, 3])
def test_lstm_output_shape(hidden, n_features):
    model = LSTMRegressor(n_features, hidden)
    x = torch.randn(7, N_STEPS, n_features)
    y = model(x)
    assert y.shape == (7,)
    assert torch.isfinite(y).all()


def test_fc_baseline_output_shape():
    model = FCBaseline(n_features=2, n_steps=N_STEPS, hidden_size=10)
    y = model(torch.randn(7, N_STEPS, 2))
    assert y.shape == (7,)


def test_fc_baseline_flattens_the_whole_window():
    model = FCBaseline(n_features=2, n_steps=N_STEPS, hidden_size=10)
    assert model.hidden.in_features == N_STEPS * 2 == 20
    assert model.hidden.out_features == 10
    assert model.out.out_features == 1


def test_sigmoid_output_is_bounded():
    model = LSTMRegressor(2, 10, output_activation="sigmoid")
    y = model(torch.randn(32, N_STEPS, 2) * 10)
    assert ((y >= 0) & (y <= 1)).all()


def test_linear_output_can_leave_the_unit_interval():
    """The reason we default to linear rather than Eq. 7's sigmoid (D-10)."""
    set_seed(0)
    model = LSTMRegressor(2, 10, output_activation="linear")
    with torch.no_grad():
        model.fc.weight.fill_(5.0)
        model.fc.bias.fill_(-3.0)
    y = model(torch.randn(64, N_STEPS, 2))
    assert (y < 0).any() or (y > 1).any()


def test_unknown_activation_raises():
    with pytest.raises(ValueError, match="output_activation"):
        LSTMRegressor(2, 10, output_activation="softmax")
    with pytest.raises(ValueError, match="activation"):
        FCBaseline(n_features=2, activation="elu")


# --- Parameter counts -----------------------------------------------------

def test_matlab_parameter_count_for_h10_f2():
    """LSTM 4*10*12 + 40 = 520, plus FC 11 -> 531 (CLAUDE.md Phase 4)."""
    assert matlab_parameter_count(n_features=2, hidden_size=10) == 531


def test_pytorch_parameter_count_exceeds_matlab_by_the_extra_bias():
    """PyTorch keeps a second bias vector, so 571 not 531 (D-11)."""
    model = LSTMRegressor(2, 10)
    assert count_parameters(model) == 571
    assert count_parameters(model) - matlab_parameter_count(2, 10) == 4 * 10


def test_freeze_bias_hh_matches_matlab_effective_count():
    model = LSTMRegressor(2, 10, freeze_bias_hh=True)
    assert count_parameters(model, trainable_only=True) == 531
    assert torch.equal(model.lstm.bias_hh_l0, torch.zeros(40))
    assert not model.lstm.bias_hh_l0.requires_grad


def test_frozen_bias_hh_stays_zero_after_an_update():
    model = LSTMRegressor(2, 10, freeze_bias_hh=True)
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.1)
    loss = ((model(torch.randn(8, N_STEPS, 2)) - torch.ones(8)) ** 2).mean()
    loss.backward()
    opt.step()
    assert torch.equal(model.lstm.bias_hh_l0, torch.zeros(40))


@pytest.mark.parametrize("hidden", [5, 10, 20])
def test_parameter_counts_across_hidden_sizes(hidden):
    model = LSTMRegressor(2, hidden, freeze_bias_hh=True)
    assert count_parameters(model, trainable_only=True) == matlab_parameter_count(2, hidden)


# --- MATLAB-style initialisation -----------------------------------------

def test_forget_gate_bias_is_one_and_others_are_zero():
    hidden = 10
    model = LSTMRegressor(2, hidden, init="matlab")
    bias_ih = model.lstm.bias_ih_l0.detach()

    forget = bias_ih[FORGET_GATE * hidden:(FORGET_GATE + 1) * hidden]
    assert torch.equal(forget, torch.ones(hidden)), "forget-gate bias must be 1"

    for gate in range(4):
        if gate == FORGET_GATE:
            continue
        block = bias_ih[gate * hidden:(gate + 1) * hidden]
        assert torch.equal(block, torch.zeros(hidden)), f"gate {gate} bias must be 0"


def test_the_unit_bias_lives_only_in_bias_ih():
    """Otherwise the two PyTorch bias vectors would sum to an effective 2."""
    model = LSTMRegressor(2, 10, init="matlab")
    assert torch.equal(model.lstm.bias_hh_l0.detach(), torch.zeros(40))
    total = model.lstm.bias_ih_l0.detach() + model.lstm.bias_hh_l0.detach()
    assert torch.equal(total[10:20], torch.ones(10))


def test_hidden_weights_are_orthogonal_per_gate():
    hidden = 10
    model = LSTMRegressor(2, hidden, init="matlab")
    w = model.lstm.weight_hh_l0.detach()
    for gate in range(4):
        block = w[gate * hidden:(gate + 1) * hidden]
        product = block @ block.T
        assert torch.allclose(product, torch.eye(hidden), atol=1e-5), (
            f"gate {gate} hidden weights are not orthogonal"
        )


def test_input_weights_use_glorot_scale():
    hidden, n_features = 20, 2
    model = LSTMRegressor(n_features, hidden, init="matlab")
    w = model.lstm.weight_ih_l0.detach()
    limit = np.sqrt(6.0 / (4 * hidden + n_features))
    assert w.abs().max().item() <= limit + 1e-6
    assert w.abs().max().item() > limit * 0.5, "weights look degenerate"


def test_fc_bias_starts_at_zero():
    model = LSTMRegressor(2, 10, init="matlab")
    assert torch.equal(model.fc.bias.detach(), torch.zeros(1))


def test_torch_default_init_differs_from_matlab():
    set_seed(0)
    a = LSTMRegressor(2, 10, init="torch_default")
    assert not torch.equal(a.lstm.bias_ih_l0[10:20].detach(), torch.ones(10))


def test_unknown_init_raises():
    with pytest.raises(ValueError, match="init"):
        LSTMRegressor(2, 10, init="he")


# --- Learning capacity ----------------------------------------------------

def _overfit(model, X, y, steps=400, lr=0.05):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss = ((model(X) - y) ** 2).mean()
        loss.backward()
        opt.step()
    return float(loss.item())


def test_lstm_overfits_a_tiny_batch():
    set_seed(0)
    X = torch.randn(8, N_STEPS, 2)
    y = torch.randn(8)
    model = LSTMRegressor(2, 20)
    before = float(((model(X) - y) ** 2).mean().item())
    after = _overfit(model, X, y)
    assert after < 1e-3, f"failed to overfit 8 samples (loss {after:.5f})"
    assert after < before


def test_fc_baseline_overfits_a_tiny_batch():
    set_seed(0)
    X = torch.randn(8, N_STEPS, 2)
    y = torch.randn(8)
    model = FCBaseline(n_features=2, n_steps=N_STEPS, hidden_size=20)
    assert _overfit(model, X, y) < 1e-3


def test_lstm_uses_the_whole_sequence():
    """Changing an early step must change the prediction."""
    set_seed(0)
    model = LSTMRegressor(2, 10)
    x = torch.randn(1, N_STEPS, 2)
    base = model(x).item()
    x2 = x.clone()
    x2[0, 0, :] += 5.0                       # perturb the oldest step
    assert abs(model(x2).item() - base) > 1e-7


# --- build_model ----------------------------------------------------------

def test_build_model_reads_config(cfg):
    model = build_model(cfg, n_features=2, hidden_size=10, kind="lstm")
    assert isinstance(model, LSTMRegressor)
    assert model.hidden_size == 10
    assert count_parameters(model) == 571          # freeze_bias_hh defaults to false

    fc = build_model(cfg, n_features=2, hidden_size=0, kind="fc")
    assert isinstance(fc, FCBaseline)
    assert fc.hidden_size == cfg["model"]["fc_baseline_hidden"]


def test_build_model_honours_freeze_bias_hh(cfg):
    frozen = build_model(with_model(cfg, freeze_bias_hh=True), 2, 10)
    assert count_parameters(frozen, trainable_only=True) == 531


def test_build_model_unknown_kind_raises(cfg):
    with pytest.raises(ValueError, match="kind"):
        build_model(cfg, 2, 10, kind="gru")


def test_set_seed_makes_initialisation_reproducible():
    set_seed(123)
    a = LSTMRegressor(2, 10)
    set_seed(123)
    b = LSTMRegressor(2, 10)
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.equal(pa, pb)
