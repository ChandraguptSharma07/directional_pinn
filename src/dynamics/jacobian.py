"""Numerical Jacobians, used to verify analytic Jacobians.

Two independent methods approximate column j of the Jacobian of f at x:

central differences   (f(x + h e_j) - f(x - h e_j)) / (2 h)
    error = truncation O(h^2 |f'''|) + rounding O(eps |f| / h), eps ~ 2.2e-16
complex step          Im f(x + i h e_j) / h,  with h = 1e-30
    no subtraction, hence no cancellation: exact up to rounding for functions
    built from operations that extend analytically to complex numbers
    (Squire & Trapp, SIAM Review 40, 1998)

The two methods share nothing except f itself, so an analytic Jacobian that
agrees with both is very unlikely to be wrong.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

VectorField = Callable[[np.ndarray], np.ndarray]


def central_difference_jacobian(f: VectorField, x: np.ndarray, h: float | None = None) -> np.ndarray:
    """Jacobian of f at x by central differences. Shape (m, d) for f from R^d to R^m.

    The default step for coordinate j is eps^(1/3) * max(1, |x_j|), which
    balances truncation and rounding error for smooth f. The difference is
    divided by the step that was actually taken after rounding x_j +- h.
    """
    x = np.asarray(x, dtype=float)
    steps = np.full(x.shape, h) if h is not None else np.finfo(float).eps ** (1 / 3) * np.maximum(1.0, np.abs(x))
    columns = []
    for j in range(x.size):
        forward, backward = x.copy(), x.copy()
        forward[j] += steps[j]
        backward[j] -= steps[j]
        columns.append((f(forward) - f(backward)) / (forward[j] - backward[j]))
    return np.column_stack(columns)


def complex_step_jacobian(f: VectorField, x: np.ndarray, h: float = 1e-30) -> np.ndarray:
    """Jacobian of f at x by the complex-step method. Shape (m, d).

    f must accept complex input and use only operations that extend
    analytically to complex numbers (no abs, comparisons or casts to float).
    """
    x = np.asarray(x, dtype=float)
    columns = []
    for j in range(x.size):
        shifted = x.astype(complex)
        shifted[j] += 1j * h
        columns.append(np.imag(f(shifted)) / h)
    return np.column_stack(columns)


NUMERICAL_JACOBIANS = {
    "complex_step": complex_step_jacobian,
    "central_difference": central_difference_jacobian,
}


@dataclass(frozen=True, eq=False)
class JacobianCheck:
    """Analytic vs numerical Jacobians at a set of states.

    errors[method][k] is the relative error |J_analytic - J_numerical|_F / |J_analytic|_F
    at states[k] (Frobenius norm). A method passes if its largest error is at
    most tolerances[method]; the check passes if every method passes.
    """

    states: np.ndarray
    errors: dict[str, np.ndarray]
    tolerances: dict[str, float]

    @property
    def passed(self) -> bool:
        return all(self.errors[method].max() <= self.tolerances[method] for method in self.errors)

    def summary(self) -> dict:
        """JSON-friendly summary of the check."""
        return {
            "n_states": len(self.states),
            "passed": self.passed,
            "methods": {
                method: {
                    "max_relative_error": float(errors.max()),
                    "median_relative_error": float(np.median(errors)),
                    "tolerance": self.tolerances[method],
                    "passed": bool(errors.max() <= self.tolerances[method]),
                }
                for method, errors in self.errors.items()
            },
        }


def check_jacobian(
    f: VectorField,
    jacobian: Callable[[np.ndarray], np.ndarray],
    states: np.ndarray,
    tolerances: dict[str, float],
) -> JacobianCheck:
    """Compare jacobian(state) with every numerical method named in `tolerances`, at each state."""
    for method in tolerances:
        if method not in NUMERICAL_JACOBIANS:
            raise ValueError(f"unknown method {method!r}; choose from {sorted(NUMERICAL_JACOBIANS)}")
    states = np.atleast_2d(np.asarray(states, dtype=float))
    analytic = [jacobian(state) for state in states]
    errors = {
        method: np.array(
            [
                np.linalg.norm(J - NUMERICAL_JACOBIANS[method](f, state)) / np.linalg.norm(J)
                for state, J in zip(states, analytic)
            ]
        )
        for method in tolerances
    }
    return JacobianCheck(states=states, errors=errors, tolerances=dict(tolerances))
