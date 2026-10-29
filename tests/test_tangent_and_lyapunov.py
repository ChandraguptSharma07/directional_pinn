"""Tests for tangent propagators and Lyapunov quantities (Phase 3).

The linear systems of Phase 0 serve as exact oracles: for constant A the step
propagator is expm(A h), the exponents are the eigenvalues, and the Phase 0
closed forms give the backward and forward vectors.
"""

import numpy as np
import pytest
from scipy.linalg import expm

from src.dynamics.lyapunov import (
    line_misalignment,
    lyapunov_exponents,
    make_continuous,
    qr_iteration,
    running_exponents,
    singular_directions,
)
from src.dynamics.tangent_dynamics import propagators_over, step_propagators
from src.solvers.reference_solver import integrate
from src.systems.linear_saddle import lyapunov_directions_2d, sheared_saddle
from src.systems.lorenz63 import LorenzParams, lorenz_divergence, lorenz_jacobian, lorenz_rhs

PARAMS = LorenzParams()
SHEARED = sheared_saddle(2.0)


def constant(A):
    return lambda t: np.broadcast_to(A, np.shape(t) + A.shape)


@pytest.fixture(scope="module")
def lorenz_run():
    """A trajectory on the attractor with dense output, and its step propagators."""
    rhs = lambda t, s: lorenz_rhs(t, s, PARAMS)
    x0 = integrate(rhs, [1.0, 1.0, 1.0], [0.0, 50.0], rtol=1e-10, atol=1e-12).y[-1]
    t = np.linspace(0.0, 60.0, 6001)
    run = integrate(rhs, x0, t, rtol=1e-12, atol=1e-14, dense_output=True)
    props = step_propagators(lambda s: lorenz_jacobian(run.at(s), PARAMS), t)
    return run, props


# --- step propagators and products ------------------------------------------------


def test_step_propagators_of_a_constant_matrix_are_matrix_exponentials():
    t = np.linspace(0.0, 1.0, 51)
    props = step_propagators(constant(SHEARED.A), t)
    np.testing.assert_allclose(props, np.broadcast_to(expm(SHEARED.A * 0.02), props.shape), rtol=1e-10, atol=1e-12)


def test_step_propagators_reject_nonuniform_grid():
    with pytest.raises(ValueError):
        step_propagators(constant(SHEARED.A), np.array([0.0, 0.1, 0.3]))


@pytest.mark.parametrize("m", [1, 2, 3, 7, 8, 13])
def test_products_match_naive_products(m):
    rng = np.random.default_rng(0)
    props = np.eye(3) + 0.1 * rng.normal(size=(20, 3, 3))
    products = propagators_over(props, m)
    assert products.shape == (20 - m + 1, 3, 3)
    for j in (0, 5, 20 - m):
        naive = np.eye(3)
        for i in range(j, j + m):
            naive = props[i] @ naive
        np.testing.assert_allclose(products[j], naive, rtol=1e-12, atol=1e-14)


def test_products_of_constant_propagators_are_matrix_exponentials():
    t = np.linspace(0.0, 2.0, 101)
    products = propagators_over(step_propagators(constant(SHEARED.A), t), 25)
    np.testing.assert_allclose(products[10], expm(SHEARED.A * 0.5), rtol=1e-9)


def test_lorenz_propagator_predicts_nearby_trajectories(lorenz_run):
    """Phi(t + dt, t) delta matches the difference of two nonlinear solutions for small delta."""
    run, props = lorenz_run
    rhs = lambda t, s: lorenz_rhs(t, s, PARAMS)
    h = run.t[1] - run.t[0]
    rng = np.random.default_rng(1)
    for start, m in ((1000, 50), (3000, 100), (4000, 25)):
        delta = 1e-6 * rng.normal(size=3)
        t0, t1 = run.t[start], run.t[start + m]
        base = integrate(rhs, run.y[start], [t0, t1], rtol=1e-13, atol=1e-15).y[-1]
        moved = integrate(rhs, run.y[start] + delta, [t0, t1], rtol=1e-13, atol=1e-15).y[-1]
        linear = propagators_over(props[start : start + m], m)[0] @ delta
        np.testing.assert_allclose(moved - base, linear, rtol=1e-4, atol=1e-10)
        assert np.isclose(m * h, t1 - t0)


# --- QR method ----------------------------------------------------------------------


def test_qr_bases_stay_orthonormal_and_reproduce_the_propagators(lorenz_run):
    _, props = lorenz_run
    Q, log_stretch = qr_iteration(props, np.eye(3))
    identity_error = np.abs(np.swapaxes(Q, 1, 2) @ Q - np.eye(3)).max()
    assert identity_error < 1e-12
    R = np.swapaxes(Q[1:], 1, 2) @ props @ Q[:-1]  # Q_{i+1}^T Phi_i Q_i is upper triangular
    assert np.abs(np.tril(R, k=-1)).max() < 1e-12
    np.testing.assert_allclose(np.log(np.diagonal(R, axis1=1, axis2=2)), log_stretch, atol=1e-12)


def test_qr_on_sheared_saddle_gives_eigenvalues_and_backward_vector():
    t = np.linspace(0.0, 30.0, 3001)
    props = step_propagators(constant(SHEARED.A), t)
    angle = 0.7
    Q0 = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    Q, log_stretch = qr_iteration(props, Q0)
    np.testing.assert_allclose(lyapunov_exponents(log_stretch, 0.01), [1.0, -1.0], atol=0.03)
    b1 = lyapunov_directions_2d(SHEARED.A)["b1"]
    assert line_misalignment(Q[-1][:, 0], b1) < 1e-10


def test_lorenz_lyapunov_exponents_are_plausible(lorenz_run):
    """Over 60 time units: one positive, one near zero, one strongly negative exponent.
    Their sum equals the constant trace -(sigma + 1 + beta) up to integration error."""
    _, props = lorenz_run
    _, log_stretch = qr_iteration(props, np.eye(3))
    exponents = lyapunov_exponents(log_stretch, 0.01)
    assert 0.6 < exponents[0] < 1.2
    assert abs(exponents[1]) < 0.15
    assert -15.0 < exponents[2] < -14.0
    assert exponents.sum() == pytest.approx(lorenz_divergence(PARAMS), abs=1e-6)
    assert running_exponents(log_stretch, 0.01).shape == log_stretch.shape


def test_random_tangent_vector_grows_at_the_largest_exponent(lorenz_run):
    _, props = lorenz_run
    _, log_stretch = qr_iteration(props, np.eye(3))
    rng = np.random.default_rng(2)
    v = rng.normal(size=3)
    log_growth = 0.0
    for P in props:  # renormalise to avoid overflow
        v = P @ v
        norm = np.linalg.norm(v)
        log_growth += np.log(norm)
        v /= norm
    assert log_growth / 60.0 == pytest.approx(lyapunov_exponents(log_stretch, 0.01)[0], abs=0.05)


def test_backward_vector_forgets_its_starting_basis(lorenz_run):
    """b1 after a spin-up does not depend on the initial orthonormal basis."""
    _, props = lorenz_run
    rng = np.random.default_rng(3)
    Qa, _ = qr_iteration(props, np.eye(3))
    Qb, _ = qr_iteration(props, np.linalg.qr(rng.normal(size=(3, 3)))[0])
    late = slice(3000, None)  # after 30 time units
    assert line_misalignment(Qa[late, :, 0], Qb[late, :, 0]).max() < 1e-8


# --- singular directions ------------------------------------------------------------


def test_singular_directions_definition():
    rng = np.random.default_rng(4)
    props = rng.normal(size=(10, 3, 3))
    sigma, V = singular_directions(props)
    assert np.all(np.diff(sigma, axis=1) <= 0)
    for P, s, v in zip(props, sigma, V):
        np.testing.assert_allclose(np.linalg.norm(P @ v, axis=0), s, rtol=1e-12)
        np.testing.assert_allclose(v.T @ v, np.eye(3), atol=1e-12)


def test_long_horizon_singular_vector_of_sheared_saddle_is_forward_vector():
    t = np.linspace(0.0, 20.0, 2001)
    product = propagators_over(step_propagators(constant(SHEARED.A), t), 2000)
    _, V = singular_directions(product)
    assert line_misalignment(V[0][:, 0], lyapunov_directions_2d(SHEARED.A)["f1"]) < 1e-8


def test_make_continuous():
    vectors = np.array([[1.0, 0.0], [-1.0, 0.1], [1.0, 0.2]])
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    out = make_continuous(vectors)
    assert np.all(np.sum(out[1:] * out[:-1], axis=1) > 0)
