"""Tests for the Lorenz reference windows and the tolerance-study helpers."""

import numpy as np
import pytest

from src.solvers.lorenz_reference import build_reference_windows, window_start_times
from src.solvers.reference_solver import integrate, setting_label, tolerance_study, trusted_until
from src.systems.lorenz63 import LorenzParams, lorenz_rhs

PARAMS = LorenzParams()
SOLVER = {"method": "DOP853", "rtol": 1e-12, "atol": 1e-14}
RHS = lambda t, state: lorenz_rhs(t, state, PARAMS)


@pytest.fixture(scope="module")
def windows():
    return build_reference_windows(
        PARAMS, seed_state=[1.0, 1.0, 1.0], transient=20.0, spinup=5.0, gap=4.0, n_windows=3,
        length=2.0, dt=0.01, solver=SOLVER,
    )


def test_window_start_times():
    np.testing.assert_allclose(window_start_times(100.0, 30.0, 20.0, 3), [130.0, 150.0, 170.0])


def test_windows_start_on_the_attractor_and_differ(windows):
    for window in windows:
        assert np.all(np.abs(window.x0[:2]) < 25.0) and 0.0 < window.x0[2] < 50.0
        np.testing.assert_array_equal(window.reference.y[0], window.x0)
    assert len({tuple(np.round(w.x0, 6)) for w in windows}) == len(windows)


def test_past_ends_at_the_initial_condition(windows):
    for window in windows:
        np.testing.assert_allclose(window.past(0.0), window.x0, atol=1e-12)
        assert window.past(np.array([-window.spinup, -1.0, 0.0])).shape == (3, 3)
    with pytest.raises(ValueError):
        windows[0].past(0.5)


def test_reference_follows_the_long_run_for_a_short_time(windows):
    """The fresh reference and the long run start at the same state; they agree until chaos separates them."""
    window = windows[1]
    t = window.reference.t[window.reference.t <= 1.0]
    continuation = window.long_run.at(window.start_in_long_run + t)
    np.testing.assert_allclose(window.reference.y[: len(t)], continuation, atol=1e-8)


def test_reference_satisfies_the_ode(windows):
    """Dense output differentiated by central differences matches the right-hand side."""
    window = windows[0]
    t = np.linspace(0.1, 1.9, 19)
    h = 1e-5
    derivative = (window.reference.at(t + h) - window.reference.at(t - h)) / (2 * h)
    np.testing.assert_allclose(derivative, lorenz_rhs(0.0, window.reference.at(t), PARAMS), rtol=1e-6, atol=1e-5)


def test_tolerance_study_errors_shrink_with_tighter_tolerance():
    t = np.linspace(0.0, 2.0, 201)
    ladder = [{"method": "DOP853", "rtol": r, "atol": r * 1e-2} for r in (1e-6, 1e-8, 1e-10)]
    _, deviations = tolerance_study(RHS, [1.0, 1.0, 20.0], t, ladder, SOLVER)
    final = [deviations[setting_label(s)][-1] for s in ladder]
    assert final[0] > final[1] > final[2]
    assert final[2] < 1e-7


def test_trusted_until():
    t = np.linspace(0.0, 1.0, 11)
    deviation = np.array([0, 1, 2, 3, 4, 50, 60, 70, 80, 90, 100]) * 1e-10
    assert trusted_until(t, deviation, 1e-9) == pytest.approx(0.4)
    assert trusted_until(t, deviation, 1.0) == pytest.approx(1.0)


def test_dense_output_requires_request():
    sol = integrate(RHS, [1.0, 1.0, 20.0], np.linspace(0.0, 0.1, 3))
    with pytest.raises(ValueError):
        sol.at(0.05)
