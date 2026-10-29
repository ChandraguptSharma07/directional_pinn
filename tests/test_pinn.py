"""Tests for the PINN model, residual and training loop (Phase 4)."""

import numpy as np
import pytest
import torch

from src.pinn.model import PINN
from src.pinn.residual import lorenz_residual, lorenz_rhs_torch, ode_residual, state_and_time_derivative
from src.pinn.train import TrainConfig, collocation_times, evaluate, train_pinn
from src.systems.lorenz63 import LorenzParams, lorenz_fixed_points, lorenz_rhs

PARAMS = LorenzParams()
torch.set_default_dtype(torch.float64)


def saddle_rhs(x):
    return torch.stack([x[:, 0], -x[:, 1]], dim=1)


def test_residual_of_exact_toy_solution_is_zero():
    """x = x0 e^t, y = y0 e^{-t} solves the linear saddle exactly."""
    exact = lambda t: torch.cat([0.3 * torch.exp(t), -1.2 * torch.exp(-t)], dim=1)
    t = torch.linspace(0.0, 2.0, 50).unsqueeze(1)
    _, _, residual = ode_residual(exact, t, saddle_rhs, create_graph=False)
    assert residual.abs().max() < 1e-12


def test_residual_of_perturbed_toy_solution_is_known():
    """Adding c t to the exact solution changes the residual to c - A (c t)."""
    c = torch.tensor([[0.5, -0.25]])
    approx = lambda t: torch.cat([0.3 * torch.exp(t), -1.2 * torch.exp(-t)], dim=1) + c * t
    t = torch.linspace(0.0, 2.0, 50).unsqueeze(1)
    _, _, residual = ode_residual(approx, t, saddle_rhs, create_graph=False)
    expected = c - torch.cat([c[:, :1] * t, -c[:, 1:] * t], dim=1)
    torch.testing.assert_close(residual, expected, rtol=1e-12, atol=1e-12)


def test_lorenz_residual_of_a_fixed_point_and_of_a_polynomial_path():
    point = torch.tensor(lorenz_fixed_points(PARAMS)["C+"]).unsqueeze(0)
    t = torch.linspace(0.0, 1.0, 20).unsqueeze(1)
    _, _, residual = lorenz_residual(lambda s: point.expand(len(s), 3) + 0.0 * s, t, PARAMS, create_graph=False)
    assert residual.abs().max() < 1e-12
    path = lambda s: torch.cat([1.0 + s, 2.0 * s**2, 3.0 - s], dim=1)  # derivative (1, 4 s, -1)
    x, dxdt, residual = lorenz_residual(path, t, PARAMS, create_graph=False)
    expected = dxdt.detach().numpy() - lorenz_rhs(0.0, x.detach().numpy(), PARAMS)
    np.testing.assert_allclose(dxdt.detach().numpy(), np.column_stack([np.ones(20), 4 * t.numpy()[:, 0], -np.ones(20)]))
    np.testing.assert_allclose(residual.detach().numpy(), expected, rtol=1e-12, atol=1e-12)


def test_torch_rhs_matches_numpy_rhs():
    rng = np.random.default_rng(0)
    states = rng.uniform(-20, 20, size=(30, 3))
    np.testing.assert_allclose(lorenz_rhs_torch(torch.tensor(states), PARAMS).numpy(),
                               lorenz_rhs(0.0, states, PARAMS), rtol=1e-14)


def test_time_derivative_of_network_matches_finite_differences():
    torch.manual_seed(0)
    model = PINN(2.0, [0.0, 0.0, 25.0], [10.0, 10.0, 10.0])
    t = torch.linspace(0.1, 1.9, 7).unsqueeze(1)
    _, dxdt = state_and_time_derivative(model, t, create_graph=False)
    h = 1e-6
    numerical = (model(t + h) - model(t - h)) / (2 * h)
    torch.testing.assert_close(dxdt, numerical.detach(), rtol=1e-6, atol=1e-6)


def test_model_scaling_and_shapes():
    torch.manual_seed(1)
    model = PINN(3.0, [1.0, 2.0, 3.0], [10.0, 20.0, 30.0], hidden=[8, 8])
    t = torch.tensor([[0.0], [1.5], [3.0]])
    raw = model.net(2.0 * t / 3.0 - 1.0)
    torch.testing.assert_close(model(t), torch.tensor([[1.0, 2.0, 3.0]]) + torch.tensor([[10.0, 20.0, 30.0]]) * raw)
    with pytest.raises(ValueError):
        PINN(1.0, [0.0], [1.0], activation="relu6")


def test_collocation_times_are_stratified():
    times = collocation_times(10, 2.0, torch.Generator().manual_seed(0), torch.float64).squeeze(1)
    bins = torch.floor(times / 0.2).long()
    assert torch.equal(bins, torch.arange(10))


SMALL = dict(t_final=0.5, x0=[1.0, 2.0, 20.0], hidden=[16, 16], iterations=30, n_collocation=64, log_every=10)


def test_training_is_deterministic_for_a_seed():
    model_a, history_a = train_pinn(TrainConfig(seed=3, **SMALL), PARAMS)
    model_b, history_b = train_pinn(TrainConfig(seed=3, **SMALL), PARAMS)
    assert history_a["loss"] == history_b["loss"]
    for p, q in zip(model_a.parameters(), model_b.parameters()):
        assert torch.equal(p, q)
    _, history_c = train_pinn(TrainConfig(seed=4, **SMALL), PARAMS)
    assert history_c["loss"] != history_a["loss"]


def test_training_reduces_the_loss():
    cfg = TrainConfig(seed=0, **{**SMALL, "iterations": 300, "lr": 3e-3, "log_every": 50})
    _, history = train_pinn(cfg, PARAMS)
    assert history["loss"][-1] < 0.5 * history["loss"][0]
    assert history["config"]["seed"] == 0 and history["runtime_seconds"] > 0


def test_evaluate_returns_float64_and_leaves_model_unchanged():
    model, _ = train_pinn(TrainConfig(seed=0, **SMALL), PARAMS)
    first_dtype = next(model.parameters()).dtype
    out = evaluate(model, np.linspace(0.0, 0.5, 11), PARAMS)
    assert out["state"].dtype == np.float64 and out["residual"].shape == (11, 3)
    assert next(model.parameters()).dtype == first_dtype
    np.testing.assert_allclose(out["residual"], out["derivative"] - lorenz_rhs(0.0, out["state"], PARAMS),
                               rtol=1e-10, atol=1e-10)
