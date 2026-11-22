"""Tests for the dial-study systems and the general-system PINN path."""

import numpy as np
import pytest
import torch

from src.dynamics.jacobian import central_difference_jacobian
from src.pinn.residual import ode_residual
from src.pinn.runs import job_system
from src.pinn.train import TrainConfig, evaluate, system_rhs, train_pinn, train_pinns
from src.solvers.reference_solver import integrate
from src.systems.dial_systems import (
    DAVIS_SKODJE_WINDOWS,
    SADDLE_WINDOWS,
    DavisSkodje,
    StiffSaddle,
    dial_settings,
    system_from_dict,
)


def test_dial_settings_cover_the_planned_rate_differences():
    dials = [system.dial for system in dial_settings()]
    assert dials == [2.0, 5.0, 17.0, 65.0, 2.0, 5.0, 17.0, 65.0]
    assert sorted(SADDLE_WINDOWS) == sorted(DAVIS_SKODJE_WINDOWS) == list(range(17))


@pytest.mark.parametrize("kappa", [1.0, 64.0])
def test_saddle_matrix_has_the_planned_eigenvalues_and_directions(kappa):
    system = StiffSaddle(kappa)
    A = system.matrix
    assert np.allclose(A, A.T)
    assert np.allclose(A @ system.unstable_direction, system.unstable_direction)
    assert np.allclose(A @ system.stable_direction, -kappa * system.stable_direction)
    assert np.isclose(system.unstable_direction @ system.stable_direction, 0.0)


@pytest.mark.parametrize("kappa", [1.0, 16.0])
def test_saddle_exact_solution_matches_the_numerical_solution(kappa):
    system = StiffSaddle(kappa)
    t = np.linspace(0.0, 2.0, 201)
    for window in (0, 4):
        numerical = integrate(system.rhs, system.initial_state(window), t, rtol=1e-12, atol=1e-14).y
        exact = system.exact_solution(t, window)
        assert np.allclose(exact[0], system.initial_state(window))
        assert np.max(np.abs(numerical - exact)) < 1e-9 * np.max(np.abs(exact)) + 1e-12


def test_davis_skodje_jacobian_matches_finite_differences():
    system = DavisSkodje(18.0)
    rng = np.random.default_rng(0)
    for y in rng.uniform([0.1, -0.5], [4.0, 1.5], size=(10, 2)):
        numeric = central_difference_jacobian(lambda z: system.rhs(0.0, z), y)
        assert np.allclose(system.jacobian(y), numeric, rtol=1e-6, atol=1e-6)
    batch = rng.uniform([0.1, -0.5], [4.0, 1.5], size=(3, 4, 2))
    assert system.jacobian(batch).shape == (3, 4, 2, 2)
    assert np.allclose(np.linalg.eigvals(system.jacobian(batch[0, 0])), [-1.0, -18.0]) or np.allclose(
        np.sort(np.linalg.eigvals(system.jacobian(batch[0, 0]))), [-18.0, -1.0]
    )


@pytest.mark.parametrize("gamma", [3.0, 66.0])
def test_davis_skodje_slow_manifold_is_invariant(gamma):
    system = DavisSkodje(gamma)
    y1 = 2.0
    t = np.linspace(0.0, 2.0, 101)
    y = integrate(system.rhs, [y1, system.slow_manifold(y1)], t, rtol=1e-12, atol=1e-14).y
    assert np.allclose(y[:, 0], y1 * np.exp(-t), rtol=1e-9)
    assert np.max(np.abs(y[:, 1] - system.slow_manifold(y[:, 0]))) < 1e-9


@pytest.mark.parametrize("system", [StiffSaddle(4.0), DavisSkodje(6.0)])
def test_torch_and_numpy_right_hand_sides_agree(system):
    x = np.random.default_rng(1).uniform(0.2, 2.0, size=(7, 2))
    torch_value = system.rhs_torch(torch.tensor(x, dtype=torch.float64)).numpy()
    assert np.allclose(torch_value, system.rhs_batch(x), rtol=1e-12, atol=1e-12)
    assert system_from_dict(system.to_dict()) == system
    assert job_system({"system": system.to_dict()}) == system


def test_residual_of_the_exact_saddle_solution_is_zero():
    system = StiffSaddle(16.0)
    c, d = SADDLE_WINDOWS[2]
    vu = torch.tensor(system.unstable_direction, dtype=torch.float64)
    vs = torch.tensor(system.stable_direction, dtype=torch.float64)

    def exact(t):
        return c * torch.exp(t) * vu + d * torch.exp(-system.kappa * t) * vs

    t = torch.linspace(0.0, 2.0, 50, dtype=torch.float64).unsqueeze(1)
    _, _, residual = ode_residual(exact, t, system_rhs(system), create_graph=False)
    assert residual.abs().max().item() < 1e-10


def _short_config(system, window, seed=0):
    x0 = system.initial_state(window).tolist()
    return TrainConfig(t_final=2.0, x0=x0, seed=seed, hidden=[16, 16], out_mean=[0.5, 0.5], out_scale=[1.0, 1.0],
                       iterations=60, n_collocation=64, lambda_ic=100.0, log_every=20)


@pytest.mark.parametrize("system", [StiffSaddle(4.0), DavisSkodje(6.0)])
def test_training_on_a_general_system_runs_and_evaluates(system):
    model, history = train_pinn(_short_config(system, 1), system)
    assert np.all(np.isfinite(history["loss"]))
    assert history["loss"][-1] < history["loss"][0]
    out = evaluate(model, np.linspace(0.0, 2.0, 11), system)
    assert out["state"].shape == out["residual"].shape == (11, 2)
    assert np.allclose(out["residual"], out["derivative"] - system.rhs_batch(out["state"]), atol=1e-10)


def test_stacked_training_matches_separate_training_on_a_general_system():
    system = DavisSkodje(18.0)
    cfg = _short_config(system, 1)
    x0s = [system.initial_state(w).tolist() for w in (1, 5)]
    stacked, _ = train_pinns(cfg, [0, 1], system, x0s=x0s)
    t = np.linspace(0.0, 2.0, 21)
    for m, (seed, x0) in enumerate(zip([0, 1], x0s)):
        alone, _ = train_pinn(TrainConfig(**{**cfg.__dict__, "seed": seed, "x0": x0}), system)
        assert np.allclose(evaluate(stacked[m], t, system)["state"], evaluate(alone, t, system)["state"],
                           rtol=1e-4, atol=1e-5)


def test_lorenz_path_is_unchanged():
    from src.systems.lorenz63 import LorenzParams, lorenz_rhs

    params = LorenzParams()
    x = np.array([[1.0, 2.0, 20.0]])
    value = system_rhs(params)(torch.tensor(x, dtype=torch.float64)).numpy()[0]
    assert np.allclose(value, lorenz_rhs(0.0, x[0], params))
    with pytest.raises(TypeError):
        system_rhs(object())


def test_integrated_propagator_matches_the_closed_form_for_a_linear_system():
    from scipy.linalg import expm

    from src.dynamics.tangent_dynamics import integrated_propagators

    A = StiffSaddle(4.0).matrix
    h, n, m = 0.002, 400, 50
    steps = np.broadcast_to(expm(A * h), (n, 2, 2)).copy()
    K = integrated_propagators(steps, m, h)
    exact = np.linalg.solve(A, expm(A * m * h) - np.eye(2))  # integral of expm(A s) over [0, m h]
    assert K.shape == (n - m + 1, 2, 2)
    assert np.allclose(K, exact, rtol=1e-5, atol=1e-7)


def test_pinn_path_propagators_reproduce_the_reference_ones_on_the_exact_path():
    from src.dynamics.tangent_dynamics import (
        frozen_jacobian_propagators,
        pinn_path_step_propagators,
        propagators_over,
        step_propagators,
    )

    system = DavisSkodje(18.0)
    t = np.round(np.arange(0.0, 0.6 + 1e-9, 0.002), 10)
    sol = integrate(system.rhs, system.initial_state(3), t, rtol=1e-12, atol=1e-14, dense_output=True)
    reference = step_propagators(lambda times: system.jacobian(sol.at(times)), t)
    estimate = pinn_path_step_propagators(t, sol.y, system.jacobian)
    assert np.allclose(propagators_over(estimate, 50), propagators_over(reference, 50), rtol=1e-6, atol=1e-8)
    saddle = StiffSaddle(16.0)
    frozen = frozen_jacobian_propagators(saddle.exact_solution(t[:5], 1), saddle.jacobian, 0.1)
    from scipy.linalg import expm

    assert np.allclose(frozen, expm(saddle.matrix * 0.1))
