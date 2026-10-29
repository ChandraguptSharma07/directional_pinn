"""Tests for src/solvers/reference_solver.py."""

import numpy as np
import pytest

from src.solvers.reference_solver import integrate


def harmonic_oscillator(t, y):
    """y'' = -y as a first-order system for (position, velocity); exact solution (cos t, -sin t)."""
    return np.array([y[1], -y[0]])


def test_output_shape_and_recorded_settings():
    t = np.linspace(0.0, 1.0, 11)
    sol = integrate(harmonic_oscillator, [1.0, 0.0], t, method="DOP853", rtol=1e-9, atol=1e-11)
    assert sol.y.shape == (11, 2)
    np.testing.assert_array_equal(sol.t, t)
    assert sol.settings == {"method": "DOP853", "rtol": 1e-9, "atol": 1e-11, "t_span": [0.0, 1.0], "n_eval": 11}
    assert sol.stats["nfev"] > 0 and sol.stats["status"] == 0


def test_matches_exact_solution():
    t = np.linspace(0.0, 20.0, 401)
    sol = integrate(harmonic_oscillator, [1.0, 0.0], t)
    np.testing.assert_allclose(sol.y, np.column_stack([np.cos(t), -np.sin(t)]), atol=1e-8)


def test_tighter_tolerance_reduces_error():
    t = np.linspace(0.0, 20.0, 401)
    exact = np.column_stack([np.cos(t), -np.sin(t)])
    errors = [
        np.abs(integrate(harmonic_oscillator, [1.0, 0.0], t, rtol=rtol, atol=rtol * 1e-2).y - exact).max()
        for rtol in (1e-5, 1e-8, 1e-11)
    ]
    assert errors[0] > errors[1] > errors[2]


def test_failure_is_reported_not_silent():
    """y' = y^2 with y(0) = 1 blows up at t = 1, so no solver can reach t = 2."""
    with pytest.raises(RuntimeError, match="failed"):
        integrate(lambda t, y: y**2, [1.0], np.linspace(0.0, 2.0, 5))
