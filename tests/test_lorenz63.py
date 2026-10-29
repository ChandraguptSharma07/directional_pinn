"""Tests for the Lorenz-63 system (src/systems/lorenz63.py)."""

import numpy as np
import pytest

from src.dynamics.jacobian import central_difference_jacobian, complex_step_jacobian
from src.solvers.reference_solver import integrate
from src.systems.lorenz63 import (
    LorenzParams,
    lorenz_divergence,
    lorenz_fixed_points,
    lorenz_jacobian,
    lorenz_rhs,
    verify_lorenz_jacobian,
)

CLASSICAL = LorenzParams()
OTHER = LorenzParams(sigma=16.0, rho=45.92, beta=4.0)  # another commonly used chaotic setting
SYMMETRY = np.diag([-1.0, -1.0, 1.0])  # (x, y, z) -> (-x, -y, z)


@pytest.fixture
def states():
    rng = np.random.default_rng(7)
    return rng.uniform([-30.0, -30.0, -5.0], [30.0, 30.0, 60.0], size=(200, 3))


def sort_by_imaginary_part(eigenvalues):
    """Order complex eigenvalues robustly: conjugate pairs differ in their imaginary parts."""
    return eigenvalues[np.lexsort((eigenvalues.real, eigenvalues.imag))]


def broken_jacobian(state, params):
    """The analytic Jacobian with one sign error (+x instead of -x in row 2, column 3)."""
    J = lorenz_jacobian(state, params)
    J[..., 1, 2] *= -1.0
    return J


# --- parameters, right-hand side, Jacobian formula ----------------------------------


def test_classical_parameters():
    assert (CLASSICAL.sigma, CLASSICAL.rho, CLASSICAL.beta) == (10.0, 28.0, 8.0 / 3.0)


def test_params_from_dict():
    assert LorenzParams.from_dict({"sigma": 10, "rho": 28, "beta": 8 / 3}) == CLASSICAL
    with pytest.raises(TypeError):
        LorenzParams.from_dict({"sigma": 10, "rho": 28, "b": 8 / 3})  # typo in a key


def test_rhs_hand_computed_value():
    # (x, y, z) = (1, 2, 3): (10 (2 - 1), 1 (28 - 3) - 2, 1 * 2 - (8/3) 3) = (10, 23, -6)
    np.testing.assert_allclose(lorenz_rhs(0.0, np.array([1.0, 2.0, 3.0]), CLASSICAL), [10.0, 23.0, -6.0], rtol=1e-15)


def test_rhs_ignores_time_and_handles_batches(states):
    single = np.array([lorenz_rhs(5.0, state, OTHER) for state in states])
    np.testing.assert_array_equal(lorenz_rhs(0.0, states, OTHER), single)


def test_jacobian_hand_computed_value():
    expected = [[-10.0, 10.0, 0.0], [25.0, -1.0, -1.0], [2.0, 1.0, -8.0 / 3.0]]
    np.testing.assert_array_equal(lorenz_jacobian(np.array([1.0, 2.0, 3.0]), CLASSICAL), expected)


def test_jacobian_handles_batches(states):
    single = np.array([lorenz_jacobian(state, OTHER) for state in states])
    np.testing.assert_array_equal(lorenz_jacobian(states, OTHER), single)


# --- the Jacobian check -------------------------------------------------------------


@pytest.mark.parametrize("params", [CLASSICAL, OTHER])
def test_jacobian_matches_complex_step_and_central_differences(params, states):
    f = lambda state: lorenz_rhs(0.0, state, params)
    for state in states:
        J = lorenz_jacobian(state, params)
        np.testing.assert_allclose(complex_step_jacobian(f, state), J, rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(central_difference_jacobian(f, state), J, rtol=1e-8, atol=1e-8)


def test_verification_gate_passes_for_analytic_jacobian():
    check = verify_lorenz_jacobian(CLASSICAL)
    assert check.passed
    assert len(check.states) == 1003  # three fixed points plus 1000 random states


def test_verification_gate_detects_a_sign_error():
    check = verify_lorenz_jacobian(CLASSICAL, jacobian=broken_jacobian)
    assert not check.passed
    assert all(errors.max() > 1e-2 for errors in check.errors.values())


def test_verification_gate_detects_hard_coded_parameters():
    """A Jacobian that always uses the classical parameters fails for other parameters."""
    check = verify_lorenz_jacobian(OTHER, jacobian=lambda state, params: lorenz_jacobian(state, CLASSICAL))
    assert not check.passed


def test_quadratic_taylor_identity(states):
    """f is quadratic, so f(x + d) - f(x) - J(x) d = f(d) - J(0) d exactly (up to rounding).
    This checks J without any step size."""
    rng = np.random.default_rng(3)
    J0 = lorenz_jacobian(np.zeros(3), CLASSICAL)
    for x in states[:50]:
        d = rng.normal(size=3)
        lhs = lorenz_rhs(0.0, x + d, CLASSICAL) - lorenz_rhs(0.0, x, CLASSICAL) - lorenz_jacobian(x, CLASSICAL) @ d
        rhs = lorenz_rhs(0.0, d, CLASSICAL) - J0 @ d
        np.testing.assert_allclose(lhs, rhs, atol=1e-11)


# --- structure: trace, symmetry, normality ------------------------------------------


def test_trace_is_minus_sigma_plus_one_plus_beta(states):
    for params in (CLASSICAL, OTHER):
        traces = np.trace(lorenz_jacobian(states, params), axis1=-2, axis2=-1)
        np.testing.assert_allclose(traces, lorenz_divergence(params), rtol=1e-15)
    assert lorenz_divergence(CLASSICAL) == pytest.approx(-41.0 / 3.0)


def test_symmetry(states):
    """(x, y, z) -> (-x, -y, z) maps solutions to solutions: f(S s) = S f(s), J(S s) = S J(s) S."""
    for state in states:
        mirrored = SYMMETRY @ state
        np.testing.assert_allclose(lorenz_rhs(0.0, mirrored, CLASSICAL), SYMMETRY @ lorenz_rhs(0.0, state, CLASSICAL))
        np.testing.assert_allclose(
            lorenz_jacobian(mirrored, CLASSICAL), SYMMETRY @ lorenz_jacobian(state, CLASSICAL) @ SYMMETRY
        )


def test_jacobian_is_not_normal():
    """J J^T != J^T J (already at the origin), so the Phase 0 sheared-saddle caveat applies."""
    J = lorenz_jacobian(np.zeros(3), CLASSICAL)
    assert not np.allclose(J @ J.T, J.T @ J)


# --- fixed points and their eigenvalues ---------------------------------------------


@pytest.mark.parametrize("params", [CLASSICAL, OTHER])
def test_fixed_points_are_equilibria(params):
    points = lorenz_fixed_points(params)
    assert set(points) == {"origin", "C+", "C-"}
    for point in points.values():
        np.testing.assert_allclose(lorenz_rhs(0.0, point, params), 0.0, atol=1e-12)


def test_only_the_origin_for_rho_below_one():
    assert set(lorenz_fixed_points(LorenzParams(rho=0.5))) == {"origin"}


def test_origin_eigenvalues_match_closed_form():
    """J(0) splits into the block [[-sigma, sigma], [rho, -1]] and the entry -beta."""
    s, r, b = CLASSICAL.sigma, CLASSICAL.rho, CLASSICAL.beta
    root = np.sqrt((s + 1) ** 2 + 4 * s * (r - 1))
    expected = np.sort([(-(s + 1) - root) / 2, -b, (-(s + 1) + root) / 2])
    eigenvalues = np.linalg.eigvals(lorenz_jacobian(np.zeros(3), CLASSICAL))
    np.testing.assert_allclose(np.sort(eigenvalues.real), expected, rtol=1e-12)
    np.testing.assert_allclose(expected, [-22.8277, -8.0 / 3.0, 11.8277], atol=1e-4)


def test_nontrivial_fixed_point_eigenvalues_match_characteristic_polynomial():
    """At C+ and C- the characteristic polynomial of J is
    lambda^3 + (sigma + beta + 1) lambda^2 + beta (sigma + rho) lambda + 2 sigma beta (rho - 1)."""
    s, r, b = CLASSICAL.sigma, CLASSICAL.rho, CLASSICAL.beta
    expected = sort_by_imaginary_part(np.roots([1.0, s + b + 1, b * (s + r), 2 * s * b * (r - 1)]))
    for name in ("C+", "C-"):
        point = lorenz_fixed_points(CLASSICAL)[name]
        eigenvalues = sort_by_imaginary_part(np.linalg.eigvals(lorenz_jacobian(point, CLASSICAL)).astype(complex))
        np.testing.assert_allclose(eigenvalues, expected, rtol=1e-10)
    # One fast contracting direction and a slowly growing spiral (order: -imag, real, +imag).
    np.testing.assert_allclose(expected.real, [0.0940, -13.8546, 0.0940], atol=1e-4)
    np.testing.assert_allclose(expected.imag, [-10.1945, 0.0, 10.1945], atol=1e-4)


def test_hopf_threshold():
    """C+ and C- lose stability at rho_H = sigma (sigma + beta + 3) / (sigma - beta - 1) ≈ 24.74."""
    s, b = 10.0, 8.0 / 3.0
    rho_h = s * (s + b + 3) / (s - b - 1)
    assert rho_h == pytest.approx(24.7368, abs=1e-4)
    for rho, stable in ((rho_h - 0.5, True), (rho_h + 0.5, False)):
        params = LorenzParams(sigma=s, rho=rho, beta=b)
        eigenvalues = np.linalg.eigvals(lorenz_jacobian(lorenz_fixed_points(params)["C+"], params))
        assert (eigenvalues.real.max() < 0) == stable


def test_integration_from_a_fixed_point_stays_there():
    c_plus = lorenz_fixed_points(CLASSICAL)["C+"]
    t = np.linspace(0.0, 5.0, 51)
    solution = integrate(lambda time, state: lorenz_rhs(time, state, CLASSICAL), c_plus, t)
    np.testing.assert_allclose(solution.y, np.tile(c_plus, (len(t), 1)), atol=1e-9)
