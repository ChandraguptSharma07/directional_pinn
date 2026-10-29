"""Tests for the numerical Jacobians in src/dynamics/jacobian.py.

They use a function whose Jacobian is known in closed form and has nothing to
do with Lorenz, so that the checker is verified independently of what it checks.
"""

import numpy as np
import pytest

from src.dynamics.jacobian import check_jacobian, central_difference_jacobian, complex_step_jacobian


def f(x):
    return np.array([np.sin(x[0]) * x[1], np.exp(x[1] * x[2]), x[0] ** 2 + x[2] ** 3])


def exact_jacobian(x):
    e = np.exp(x[1] * x[2])
    return np.array(
        [
            [np.cos(x[0]) * x[1], np.sin(x[0]), 0.0],
            [0.0, x[2] * e, x[1] * e],
            [2 * x[0], 0.0, 3 * x[2] ** 2],
        ]
    )


POINTS = np.array([[0.3, -1.2, 0.7], [2.0, 0.5, -0.4], [-1.1, 1.3, 0.2]])
TOLERANCES = {"complex_step": 1e-12, "central_difference": 1e-6}


def test_complex_step_is_exact_up_to_rounding():
    for x in POINTS:
        np.testing.assert_allclose(complex_step_jacobian(f, x), exact_jacobian(x), rtol=1e-14, atol=1e-15)


def test_central_differences_are_accurate_with_default_step():
    for x in POINTS:
        np.testing.assert_allclose(central_difference_jacobian(f, x), exact_jacobian(x), rtol=1e-8, atol=1e-9)


def test_central_difference_error_is_second_order_in_the_step():
    """Halving h divides the truncation error by about 4."""
    x = POINTS[0]
    errors = [np.abs(central_difference_jacobian(f, x, h=h) - exact_jacobian(x)).max() for h in (1e-2, 5e-3)]
    assert errors[0] / errors[1] == pytest.approx(4.0, rel=0.05)


def test_check_reports_pass_and_fail():
    good = check_jacobian(f, exact_jacobian, POINTS, TOLERANCES)
    assert good.passed
    summary = good.summary()
    assert summary["n_states"] == 3 and summary["passed"]
    assert set(summary["methods"]) == set(TOLERANCES)
    bad = check_jacobian(f, lambda x: exact_jacobian(x) + 1e-3, POINTS, TOLERANCES)
    assert not bad.passed


def test_check_rejects_unknown_method():
    with pytest.raises(ValueError):
        check_jacobian(f, exact_jacobian, POINTS, {"forward_difference": 1e-6})
