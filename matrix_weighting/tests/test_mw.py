"""Tests for the matrix-weighting study code (run from matrix_weighting/: python -m pytest tests)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.integrate import solve_ivp
from scipy.linalg import expm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mw import weights as wt  # noqa: E402
from mw.model import BatchedPINN, init_seed  # noqa: E402
from mw.reference import reference  # noqa: E402
from mw.systems import DavisSkodje, LinearSaddle, Lorenz63, Lorenz96, Rossler  # noqa: E402
from mw.train import Settings, train_batch  # noqa: E402

SYSTEMS = [LinearSaddle(16.0), DavisSkodje(18.0), Lorenz63(), Lorenz63(k=4), Rossler(), Lorenz96(8)]


def sample_states(system, m=7, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(m, system.dim))
    if isinstance(system, Lorenz63):
        x[:, :3] = x[:, :3] * 8 + np.array([0, 0, 25])
    if isinstance(system, DavisSkodje):
        x[:, 0] = np.abs(x[:, 0]) + 0.1  # stay away from the pole at y1 = -1
    return x


# ------------------------------------------------------------------ systems
@pytest.mark.parametrize("system", SYSTEMS, ids=lambda s: s.name)
def test_jacobian_matches_finite_differences(system):
    x = sample_states(system)
    J = system.jac(x)
    eps = 1e-6
    for i in range(system.dim):
        d = np.zeros(system.dim)
        d[i] = eps
        fd = (system.rhs(x + d) - system.rhs(x - d)) / (2 * eps)
        np.testing.assert_allclose(J[..., :, i], fd, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("system", SYSTEMS, ids=lambda s: s.name)
def test_torch_and_numpy_agree(system):
    x = sample_states(system)
    xt = torch.tensor(x, dtype=torch.float64)
    np.testing.assert_allclose(system.rhs(xt).numpy(), system.rhs(x), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(system.jac(xt).numpy(), system.jac(x), rtol=1e-12, atol=1e-12)


def test_saddle_reference_matches_ode_solver():
    s = LinearSaddle(4.0)
    t = np.linspace(0, 2, 51)
    x0 = s.initial_state(3)
    num = solve_ivp(lambda _t, x: s.rhs(x), (0, 2), x0, method="DOP853", rtol=1e-12, atol=1e-12, t_eval=t).y.T
    np.testing.assert_allclose(reference(s, x0, t), num, rtol=1e-8, atol=1e-9)
    np.testing.assert_allclose(reference(s, x0, t), s.exact(t, 3), rtol=1e-12, atol=1e-12)


# ------------------------------------------------------------------ model
def test_time_derivative_matches_autograd_and_initial_state_is_exact():
    x0 = np.array([[1.0, 2.0, 3.0], [-1.0, 0.5, 20.0]])
    m = BatchedPINN(x0, 1.5, [0, 0, 25], [10, 10, 10], [1, 2], hidden=(16, 16), dtype=torch.float64)
    t = torch.linspace(0, 1.5, 33, dtype=torch.float64)
    x, dx = m(t)
    np.testing.assert_allclose(x[:, 0].detach().numpy(), x0, rtol=0, atol=1e-14)
    tt = t.clone().requires_grad_(True)
    xa, _ = m(tt)
    for b in range(2):
        for i in range(3):
            g = torch.autograd.grad(xa[b, :, i].sum(), tt, retain_graph=True)[0]
            np.testing.assert_allclose(dx[b, :, i].detach().numpy(), g.numpy(), rtol=1e-10, atol=1e-10)


def test_initialization_does_not_depend_on_batch_companions():
    x0 = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    alone = BatchedPINN(x0[1:2], 2.0, [0, 0], [1, 1], [init_seed(2, 7)], hidden=(8, 8))
    batch = BatchedPINN(x0, 2.0, [0, 0], [1, 1], [init_seed(0, 1), init_seed(2, 7), init_seed(1, 3)], hidden=(8, 8))
    for Wa, Wb in zip(alone.weights, batch.weights):
        torch.testing.assert_close(Wa[0], Wb[1])


def test_batched_training_equals_separate_training():
    s = LinearSaddle(4.0)
    x0 = np.stack([s.initial_state(1), s.initial_state(2)])
    st = Settings(t_final=1.0, hidden=[8, 8], n_grid=33, iterations=30, n_eval=21, dtype="float64", log_every=10)
    seeds = [init_seed(0, 1), init_seed(0, 2)]
    both = train_batch(s, "M-orc", x0, seeds, [0, 0], [2, 2], st,
                       ref_grid=np.stack([reference(s, x, np.linspace(0, 1, 33)) for x in x0]))
    for b in range(2):
        one = train_batch(s, "M-orc", x0[b:b + 1], seeds[b:b + 1], [0, 0], [2, 2], st,
                          ref_grid=reference(s, x0[b], np.linspace(0, 1, 33))[None])
        np.testing.assert_allclose(both["x_eval"][b], one["x_eval"][0], rtol=1e-9, atol=1e-11)


# ------------------------------------------------------------------ weights
def test_gramian_matches_closed_form_for_scalar_growth():
    a, T, N = 0.9, 2.0, 401
    h = T / (N - 1)
    J = torch.full((1, N, 1, 1), a, dtype=torch.float64)
    G = wt.window_gramian(wt.step_propagators(J, h), h)[0, :, 0, 0].numpy()
    t = np.linspace(0, T, N)
    exact = (np.exp(2 * a * (T - t)) - 1) / (2 * a)
    np.testing.assert_allclose(G, exact, rtol=2e-5, atol=1e-12)


def test_gramian_and_cauchy_green_match_matrix_exponential_for_saddle():
    s = LinearSaddle(16.0)
    T, N = 1.0, 201
    h = T / (N - 1)
    J = torch.tensor(np.broadcast_to(s.A, (1, N, 2, 2)).copy())
    Phi = wt.step_propagators(J, h)
    G = wt.window_gramian(Phi, h)[0].numpy()
    C = wt.final_cauchy_green(Phi)[0].numpy()
    t = np.linspace(0, T, N)
    for i in (0, 50, 150, N - 1):
        P = expm(s.A * (T - t[i]))
        np.testing.assert_allclose(C[i], P.T @ P, rtol=1e-10, atol=1e-12)
        taus = np.linspace(0, T - t[i], 4001)
        integrand = np.array([expm(s.A * u).T @ expm(s.A * u) for u in taus])
        direct = np.trapezoid(integrand, taus, axis=0) if len(taus) > 1 else np.zeros((2, 2))
        np.testing.assert_allclose(G[i], direct, rtol=2e-3, atol=1e-6)


def test_weight_normalizations():
    rng = np.random.default_rng(1)
    A = rng.normal(size=(3, 50, 4, 4))
    G = torch.tensor(A @ np.swapaxes(A, -1, -2))
    W = wt.minimax_matrix(G)
    torch.testing.assert_close(W.diagonal(dim1=-2, dim2=-1).sum(-1).mean(1), torch.full((3,), 4.0, dtype=torch.float64))
    rho = wt.best_scalar(G)
    torch.testing.assert_close(rho.mean(1), torch.ones(3, dtype=torch.float64))
    r = torch.tensor(rng.normal(size=(3, 50, 4)))
    eye = torch.eye(4, dtype=torch.float64).expand(3, 50, 4, 4)
    torch.testing.assert_close(wt.weighted_loss(r, "matrix", eye), wt.weighted_loss(r, None, None))


def test_trace_inequality_behind_the_minimax_weight():
    """lambda_max(M^-1/2 G M^-1/2) >= tr G / tr M, with equality at M proportional to G."""
    rng = np.random.default_rng(2)
    for _ in range(200):
        a, b = rng.normal(size=(2, 5, 5))
        G, M = a @ a.T + 1e-3 * np.eye(5), b @ b.T + 1e-3 * np.eye(5)
        L = np.linalg.cholesky(M)
        Li = np.linalg.inv(L)
        lam = np.linalg.eigvalsh(Li @ G @ Li.T)[-1]
        assert lam >= np.trace(G) / np.trace(M) * (1 - 1e-10)
    G = a @ a.T + np.eye(5)
    Li = np.linalg.inv(np.linalg.cholesky(3.0 * G))
    np.testing.assert_allclose(np.linalg.eigvalsh(Li @ G @ Li.T)[-1], np.trace(G) / np.trace(3.0 * G), rtol=1e-10)


def test_turinici_matches_repository_port():
    sys.path.insert(0, str(ROOT.parent))
    from src.pinn.train import turinici_time_weights
    from src.systems.lorenz63 import LorenzParams

    rng = np.random.default_rng(3)
    x = torch.tensor(rng.normal(size=(2, 64, 3)) * 8 + np.array([0, 0, 25]))
    ours = wt.turinici(x, Lorenz63().rhs, 1.0 / 63)
    port = turinici_time_weights(x, 1.0, LorenzParams()) * 64
    torch.testing.assert_close(ours, port)


@pytest.mark.parametrize("method", wt.METHODS)
def test_every_method_trains_briefly(method):
    L = Lorenz63()
    x0 = np.array([[1.0, 1.0, 1.0], [-5.0, -6.0, 22.0]])
    st = Settings(t_final=0.3, hidden=[16, 16], n_grid=40, iterations=20, n_eval=31, log_every=10, weight_every=5)
    tg = np.linspace(0, 0.3, 40)
    res = train_batch(L, method, x0, [1, 2], [0, 0, 25], [10, 10, 10], st,
                      ref_grid=np.stack([reference(L, x, tg) for x in x0]),
                      ref_eval=np.stack([reference(L, x, np.linspace(0, 0.3, 31)) for x in x0]))
    assert np.all(np.isfinite(res["metrics"]["E_rms"]))
    assert len(res["history"]["iteration"]) == 3


def test_lorenz96_matches_its_definition():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(4, 8)) * 3
    f = Lorenz96(8).rhs(x)
    for i in range(8):
        expected = (x[:, (i + 1) % 8] - x[:, (i - 2) % 8]) * x[:, (i - 1) % 8] - x[:, i] + 8.0
        np.testing.assert_allclose(f[:, i], expected, rtol=1e-12)


def test_direction_only_weight_has_equal_total_at_every_time():
    rng = np.random.default_rng(6)
    A = rng.normal(size=(2, 30, 3, 3))
    G = torch.tensor(A @ np.swapaxes(A, -1, -2))
    W = wt.direction_only(G)
    tr = W.diagonal(dim1=-2, dim2=-1).sum(-1)
    torch.testing.assert_close(tr, torch.full_like(tr, 3.0))


def test_soft_initial_condition_trains_and_reports_initial_error():
    s = LinearSaddle(16.0)
    x0 = np.stack([s.initial_state(1), s.initial_state(2)])
    tg = np.linspace(0, 1, 33)
    st = Settings(t_final=1.0, hidden=[8, 8], n_grid=33, iterations=40, n_eval=21, log_every=20, lambda_ic=100.0)
    res = train_batch(s, "M-orc", x0, [1, 2], [0, 0], [2, 2], st,
                      ref_grid=np.stack([reference(s, x, tg) for x in x0]),
                      ref_eval=np.stack([reference(s, x, np.linspace(0, 1, 21)) for x in x0]))
    ic = np.array(res["history"]["ic_error"])
    assert ic.shape == (3, 2) and np.all(ic[0] > 0) and np.all(ic[-1] < ic[0])
    m = BatchedPINN(x0, 1.0, [0, 0], [2, 2], [1, 2], hidden=(8,), hard_ic=False)
    t = torch.linspace(0, 1, 9)
    x, dx = m(t)
    assert not torch.allclose(x[:, 0], torch.tensor(x0, dtype=x.dtype))


def test_checkpoint_resume_reproduces_the_evaluation(tmp_path):
    s = LinearSaddle(4.0)
    x0 = np.stack([s.initial_state(1), s.initial_state(2)])
    st = Settings(t_final=1.0, hidden=[8, 8], n_grid=33, iterations=25, n_eval=601, log_every=10)
    ck = tmp_path / "model.pt"
    first = train_batch(s, "U", x0, [1, 2], [0, 0], [2, 2], st, checkpoint=ck)
    assert ck.exists() and not first["resumed_from_checkpoint"]
    again = train_batch(s, "U", x0, [1, 2], [0, 0], [2, 2], st, checkpoint=ck)
    assert again["resumed_from_checkpoint"]
    np.testing.assert_array_equal(first["x_eval"], again["x_eval"])
    assert first["history"] == again["history"]
