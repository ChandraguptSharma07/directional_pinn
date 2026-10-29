"""Tests for the Phase 0 linear systems (src/systems/linear_saddle.py)."""

import numpy as np
import pytest

from src.dynamics.jacobian import central_difference_jacobian
from src.solvers.reference_solver import integrate
from src.systems.linear_saddle import (
    amplification,
    approximate_solution_with_constant_residual,
    direction_angle,
    isotropic,
    leading_singular_direction,
    lyapunov_directions_2d,
    saddle,
    saddle_exact_solution,
    saddle_forced_error,
    sheared_saddle,
    sheared_saddle_exact_solution,
    unit_vectors,
)

SOLVER = {"method": "DOP853", "rtol": 1e-10, "atol": 1e-12}
Z0 = np.array([0.01, 1.0])
EPS = 1e-3
HORIZONS = [0.05, 0.1, 0.25, 0.5, 1.0]


def line_misalignment(u, v):
    """sin of the angle between the lines spanned by unit vectors u and v."""
    return abs(u[0] * v[1] - u[1] * v[0])


# --- the saddle and its solution --------------------------------------------------


def test_rhs_matches_equations():
    rng = np.random.default_rng(0)
    for z in rng.normal(size=(5, 2)):
        np.testing.assert_array_equal(saddle().rhs(0.0, z), [z[0], -z[1]])


def test_jacobian_matches_central_differences():
    system = saddle()
    z = np.array([0.3, -0.7])
    numerical = central_difference_jacobian(lambda state: system.rhs(0.0, state), z)
    np.testing.assert_allclose(system.jacobian(z), numerical, atol=1e-9)


def test_exact_solution_satisfies_ode_and_initial_condition():
    t = np.linspace(0.0, 3.0, 13)
    h = 1e-5
    z = saddle_exact_solution(t, Z0)
    dzdt = (saddle_exact_solution(t + h, Z0) - saddle_exact_solution(t - h, Z0)) / (2 * h)
    np.testing.assert_array_equal(z[0], Z0)
    np.testing.assert_allclose(dzdt, z @ saddle().A.T, rtol=1e-8)


def test_matrix_exponential_matches_closed_forms():
    t = np.linspace(0.0, 4.0, 9)
    z0 = np.array([0.3, -0.8])
    np.testing.assert_allclose(saddle().solution(t, z0), saddle_exact_solution(t, z0), rtol=1e-12)
    c = 2.0
    np.testing.assert_allclose(
        sheared_saddle(c).solution(t, z0), sheared_saddle_exact_solution(t, z0, c), rtol=1e-12, atol=1e-13
    )


def test_numerical_solution_matches_exact_solution():
    t = np.linspace(0.0, 5.0, 501)
    numerical = integrate(saddle().rhs, Z0, t, **SOLVER)
    exact = saddle_exact_solution(t, Z0)
    relative = np.linalg.norm(numerical.y - exact, axis=1) / np.linalg.norm(exact, axis=1)
    assert relative.max() < 1e-8


# --- equal-size perturbations and residuals ---------------------------------------


@pytest.mark.parametrize("direction, rate", [((1.0, 0.0), 1.0), ((0.0, 1.0), -1.0)])
def test_equal_size_state_perturbations_grow_or_shrink_exponentially(direction, rate):
    """A perturbation eps*u along the unstable (stable) axis has size eps*e^t (eps*e^-t)."""
    t = np.linspace(0.0, 5.0, 101)
    system = saddle()
    reference = integrate(system.rhs, Z0, t, **SOLVER)
    perturbed = integrate(system.rhs, Z0 + EPS * np.array(direction), t, **SOLVER)
    size = np.linalg.norm(perturbed.y - reference.y, axis=1)
    np.testing.assert_allclose(size, EPS * np.exp(rate * t), rtol=1e-3)


def test_size_ratio_of_equal_size_perturbations_is_exp_2t():
    t = np.linspace(0.0, 5.0, 11)
    system = saddle()
    unstable = np.array([system.propagator(ti) @ [EPS, 0.0] for ti in t])
    stable = np.array([system.propagator(ti) @ [0.0, EPS] for ti in t])
    ratio = np.linalg.norm(unstable, axis=1) / np.linalg.norm(stable, axis=1)
    np.testing.assert_allclose(ratio, np.exp(2 * t), rtol=1e-12)


@pytest.mark.parametrize("r", [(EPS, 0.0), (0.0, EPS), (0.6 * EPS, -0.8 * EPS)])
def test_error_from_constant_residual_matches_duhamel_formula(r):
    r = np.array(r)
    t = np.linspace(0.0, 5.0, 201)
    system = saddle()
    reference = integrate(system.rhs, Z0, t, **SOLVER)
    approx = integrate(lambda s, z: system.rhs(s, z) + r, Z0, t, **SOLVER)
    np.testing.assert_allclose(approx.y - reference.y, saddle_forced_error(t, r), rtol=1e-6, atol=1e-8)
    general = np.array([system.forced_error_operator(ti) @ r for ti in t])
    np.testing.assert_allclose(general, saddle_forced_error(t, r), rtol=1e-12, atol=1e-18)


def test_exact_solution_has_negligible_residual():
    t = np.linspace(0.0, 5.0, 5001)
    residual = saddle().residual(t, saddle_exact_solution(t, Z0))
    assert np.abs(residual).max() < 1e-6  # only the O(dt^2) finite-difference error remains


@pytest.mark.parametrize("r", [(EPS, 0.0), (0.0, EPS)])
def test_measured_residual_recovers_prescribed_constant_residual(r):
    r = np.array(r)
    t = np.linspace(0.0, 5.0, 5001)
    system = saddle()
    z_hat = approximate_solution_with_constant_residual(system, t, Z0, r)
    np.testing.assert_allclose(z_hat[0], Z0)  # starts exactly at the initial condition
    residual = system.residual(t, z_hat)
    assert np.linalg.norm(residual - r, axis=1).max() < 1e-3 * EPS


# --- orientation at fixed size, and the isotropic control -------------------------


@pytest.mark.parametrize("horizon", HORIZONS)
def test_error_increases_with_alignment_at_fixed_size(horizon):
    """Rotating a unit perturbation/residual from the x axis (alignment 1) to the
    y axis (alignment 0) strictly decreases the error after the horizon."""
    system = saddle()
    directions = unit_vectors(np.linspace(0.0, np.pi / 2, 91))
    state = amplification(system.propagator(horizon), directions)
    forced = amplification(system.forced_error_operator(horizon), directions)
    assert np.all(np.diff(state) < 0) and np.all(np.diff(forced) < 0)
    np.testing.assert_allclose(state[0] / state[-1], np.exp(2 * horizon), rtol=1e-12)
    np.testing.assert_allclose(forced[0] / forced[-1], np.exp(horizon), rtol=1e-12)


def test_isotropic_control_orientation_carries_no_information():
    system = isotropic(1.0)
    directions = unit_vectors(np.linspace(0.0, np.pi, 36, endpoint=False))
    for horizon in (0.1, 1.0, 3.0):
        state = amplification(system.propagator(horizon), directions)
        forced = amplification(system.forced_error_operator(horizon), directions)
        np.testing.assert_allclose(state, np.exp(horizon), rtol=1e-12)
        np.testing.assert_allclose(forced, np.expm1(horizon), rtol=1e-12)
    with pytest.raises(ValueError):
        lyapunov_directions_2d(system.A)  # no distinguished direction exists


# --- which "unstable direction"? (sheared, non-normal saddle) ---------------------


def test_saddle_directions_all_coincide():
    """For the symmetric saddle every notion of 'unstable direction' is the x axis."""
    d = lyapunov_directions_2d(saddle().A)
    for key in ("clv1", "b1", "f1"):
        assert line_misalignment(d[key], [1.0, 0.0]) < 1e-12
    for key in ("clv2", "b2", "f2"):
        assert line_misalignment(d[key], [0.0, 1.0]) < 1e-12


@pytest.mark.parametrize("c", [0.5, 2.0, 4.0])
def test_sheared_saddle_named_directions(c):
    d = lyapunov_directions_2d(sheared_saddle(c).A)
    n = np.hypot(2.0, c)
    assert line_misalignment(d["clv1"], [1.0, 0.0]) < 1e-12
    assert line_misalignment(d["clv2"], [-c / n, 2.0 / n]) < 1e-12
    assert line_misalignment(d["b2"], [0.0, 1.0]) < 1e-12
    assert line_misalignment(d["f1"], [2.0 / n, c / n]) < 1e-12
    np.testing.assert_array_equal(d["b1"], d["clv1"])
    np.testing.assert_array_equal(d["f2"], d["clv2"])
    for v in d.values():
        assert abs(np.linalg.norm(v) - 1.0) < 1e-12
        assert 0.0 <= direction_angle(v) < 180.0


def test_repeated_qr_converges_to_backward_vector_and_lyapunov_exponents():
    """Propagate an orthonormal basis and re-orthonormalise it with QR after every step.
    The first basis vector converges to clv1, and the time-averaged log stretching
    factors converge to the eigenvalues (+1, -1), which here are the Lyapunov exponents."""
    system = sheared_saddle(2.0)
    step, n_steps, angle = 0.5, 40, 0.5
    q = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    log_stretch = np.zeros(2)
    for _ in range(n_steps):
        q, r = np.linalg.qr(system.propagator(step) @ q)
        log_stretch += np.log(np.abs(np.diag(r)))
    assert line_misalignment(q[:, 0], [1.0, 0.0]) < 1e-12
    np.testing.assert_allclose(log_stretch / (step * n_steps), [1.0, -1.0], atol=0.05)


def test_forward_vectors_are_singular_vectors_of_long_propagator():
    system = sheared_saddle(2.0)
    d = lyapunov_directions_2d(system.A)
    _, _, vt = np.linalg.svd(system.propagator(20.0))
    assert line_misalignment(vt[0], d["f1"]) < 1e-10
    assert line_misalignment(vt[1], d["f2"]) < 1e-10


def test_perturbation_perpendicular_to_unstable_eigenvector_still_grows():
    """With shear c, a unit vector along b2 (perpendicular to the unstable eigenvector)
    eventually grows like (c/2) e^t; only the stable eigenvector decays."""
    c, tau = 2.0, 10.0
    system = sheared_saddle(c)
    d = lyapunov_directions_2d(system.A)
    phi = system.propagator(tau)
    growth = {key: amplification(phi, d[key][None, :])[0] for key in ("b1", "b2", "f1")}
    np.testing.assert_allclose(growth["b1"], np.exp(tau), rtol=1e-12)
    np.testing.assert_allclose(growth["b2"] / np.exp(tau), c / 2, rtol=1e-6)
    np.testing.assert_allclose(growth["f1"] / np.exp(tau), np.hypot(2.0, c) / 2, rtol=1e-6)
    decay = amplification(system.propagator(5.0), d["f2"][None, :])[0]
    np.testing.assert_allclose(decay, np.exp(-5.0), rtol=1e-8)


@pytest.mark.parametrize("horizon", [0.25, 1.0, 4.0])
def test_growth_is_determined_by_alignment_with_leading_singular_vector(horizon):
    """In 2-D, |Phi v|^2 = sigma_2^2 + (sigma_1^2 - sigma_2^2) (v . s_1)^2 for every unit v."""
    phi = sheared_saddle(2.0).propagator(horizon)
    directions = unit_vectors(np.linspace(0.0, np.pi, 360, endpoint=False))
    sigma = np.linalg.svd(phi, compute_uv=False)
    sigma1, s1 = leading_singular_direction(phi)
    assert sigma1 == pytest.approx(sigma[0])
    predicted = np.sqrt(sigma[1] ** 2 + (sigma[0] ** 2 - sigma[1] ** 2) * (directions @ s1) ** 2)
    np.testing.assert_allclose(amplification(phi, directions), predicted, rtol=1e-10)


def test_alignment_with_unstable_eigenvector_does_not_determine_growth():
    """Directions at +40 and -40 degrees have the same alignment |cos 40°| with the
    unstable eigenvector of the sheared saddle but grow by very different amounts."""
    phi = sheared_saddle(2.0).propagator(1.0)
    growth = amplification(phi, unit_vectors(np.radians([40.0, -40.0])))
    assert growth[0] / growth[1] > 2.0
