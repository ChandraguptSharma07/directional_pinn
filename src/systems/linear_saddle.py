"""Linear test systems  dz/dt = A z  used to build intuition in Phase 0.

The main system is the linear saddle

    dx/dt =  x
    dy/dt = -y          (A = diag(1, -1)),

with exact solution x(t) = x0 e^t, y(t) = y0 e^{-t}. The x axis is the unstable
direction (perturbations along it grow like e^t); the y axis is the stable
direction (perturbations along it shrink like e^{-t}).

Two variants act as controls:

* ``isotropic(rate)``: A = rate * I. Every direction grows at the same rate, so
  the orientation of a perturbation cannot matter. An orientation-based error
  indicator must show no advantage here.
* ``sheared_saddle(c)``: A = [[1, c], [0, -1]]. Same eigenvalues (+1, -1) as the
  saddle, but the eigenvectors are not perpendicular (A is non-normal), so
  different definitions of "the unstable direction" disagree.

All systems here are linear with constant coefficients, so every error formula
below is exact. For a nonlinear system such as Lorenz-63 the same formulas hold
only to first order in the error, with A replaced by the Jacobian evaluated
along the trajectory.

Array conventions: a state has shape (2,), a trajectory has shape (N, 2) with
row i the state at time t[i], and a set of K directions has shape (K, 2).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm


@dataclass(frozen=True, eq=False)
class LinearSystem:
    """Autonomous linear ODE  dz/dt = A z  with a constant matrix A."""

    A: np.ndarray
    name: str

    def rhs(self, t: float, z: np.ndarray) -> np.ndarray:
        """Right-hand side f(t, z) = A z, in the (t, z) form that solve_ivp expects."""
        return self.A @ z

    def jacobian(self, z: np.ndarray | None = None) -> np.ndarray:
        """Jacobian matrix Df(z) = A. It is the same at every state because f is linear."""
        return self.A.copy()

    def propagator(self, tau: float) -> np.ndarray:
        """Propagator Phi(tau) = exp(A tau).

        A perturbation delta added to the state at any time s evolves into
        Phi(tau) delta at time s + tau. This is exact because the system is
        linear, and independent of s because A does not change with time.
        """
        return expm(self.A * tau)

    def solution(self, t: np.ndarray, z0: np.ndarray) -> np.ndarray:
        """Exact solution z(t) = exp(A t) z0 at each time in t. Shape (N, 2)."""
        z0 = np.asarray(z0, dtype=float)
        return np.stack([self.propagator(ti) @ z0 for ti in np.atleast_1d(t)])

    def forced_error_operator(self, tau: float) -> np.ndarray:
        """Psi(tau) = integral_0^tau exp(A s) ds = A^{-1} (exp(A tau) - I).

        If an approximate solution has a constant residual r during a window of
        length tau, its error e obeys de/dt = A e + r there, and Duhamel's
        formula (variation of constants) gives at the end of the window

            e(end) = Phi(tau) e(start) + Psi(tau) r.

        Requires A to be invertible, which holds for every system in this module.
        """
        identity = np.eye(self.A.shape[0])
        return np.linalg.solve(self.A, self.propagator(tau) - identity)

    def residual(self, t: np.ndarray, z_hat: np.ndarray) -> np.ndarray:
        """Residual r(t) = dz_hat/dt - A z_hat of an approximate solution sampled at times t.

        The derivative uses second-order finite differences (np.gradient), so the
        result carries an O(dt^2) discretisation error. A PINN evaluates the same
        quantity with automatic differentiation instead.
        """
        dzdt = np.gradient(z_hat, t, axis=0, edge_order=2)
        return dzdt - z_hat @ self.A.T


def saddle() -> LinearSystem:
    """The linear saddle dx/dt = x, dy/dt = -y."""
    return LinearSystem(A=np.diag([1.0, -1.0]), name="saddle")


def isotropic(rate: float = 1.0) -> LinearSystem:
    """dz/dt = rate * z: the same rate in every direction (negative control)."""
    return LinearSystem(A=rate * np.eye(2), name="isotropic")


def sheared_saddle(c: float) -> LinearSystem:
    """dx/dt = x + c y, dy/dt = -y: eigenvalues +1 and -1 with non-perpendicular eigenvectors."""
    return LinearSystem(A=np.array([[1.0, c], [0.0, -1.0]]), name="sheared_saddle")


def saddle_exact_solution(t: np.ndarray, z0: np.ndarray) -> np.ndarray:
    """Closed-form saddle solution x(t) = x0 e^t, y(t) = y0 e^{-t}. Shape (N, 2)."""
    t = np.atleast_1d(np.asarray(t, dtype=float))
    x0, y0 = z0
    return np.column_stack([x0 * np.exp(t), y0 * np.exp(-t)])


def saddle_forced_error(t: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Error of a saddle approximate solution with constant residual r and exact initial state.

    Solving de/dt = A e + r, e(0) = 0, one component at a time:

        e_x(t) = r_x (e^t - 1),      e_y(t) = r_y (1 - e^{-t}).

    The unstable component grows without bound; the stable one levels off at r_y.
    """
    t = np.atleast_1d(np.asarray(t, dtype=float))
    r_x, r_y = r
    return np.column_stack([r_x * np.expm1(t), -r_y * np.expm1(-t)])


def sheared_saddle_exact_solution(t: np.ndarray, z0: np.ndarray, c: float) -> np.ndarray:
    """Closed-form solution of dx/dt = x + c y, dy/dt = -y. Shape (N, 2).

    y(t) = y0 e^{-t} does not depend on x. Substituting it into the x equation
    gives d/dt (x e^{-t}) = c y0 e^{-2t}, hence x(t) = x0 e^t + c y0 sinh(t).
    """
    t = np.atleast_1d(np.asarray(t, dtype=float))
    x0, y0 = z0
    return np.column_stack([x0 * np.exp(t) + c * y0 * np.sinh(t), y0 * np.exp(-t)])


def approximate_solution_with_constant_residual(
    system: LinearSystem, t: np.ndarray, z0: np.ndarray, r: np.ndarray
) -> np.ndarray:
    """A synthetic approximate solution z_hat(t) = z(t) + Psi(t) r. Shape (N, 2).

    It starts exactly at z0, and its residual equals the constant vector r at
    every time, because

        dz_hat/dt = A z + exp(A t) r   and   A z_hat = A z + (exp(A t) - I) r.

    Its error z_hat - z is Psi(t) r, the Duhamel integral of the residual.
    """
    r = np.asarray(r, dtype=float)
    z = system.solution(t, z0)
    error = np.stack([system.forced_error_operator(ti) @ r for ti in np.atleast_1d(t)])
    return z + error


def unit_vectors(angles: np.ndarray) -> np.ndarray:
    """Unit vectors (cos theta, sin theta) for angles in radians. Shape (K, 2)."""
    angles = np.atleast_1d(np.asarray(angles, dtype=float))
    return np.column_stack([np.cos(angles), np.sin(angles)])


def direction_angle(v: np.ndarray) -> float:
    """Angle in degrees, in [0, 180), of the line spanned by v (the sign of v is irrelevant)."""
    angle = float(np.degrees(np.arctan2(v[1], v[0])) % 180.0)
    return 0.0 if np.isclose(angle, 180.0, rtol=0.0, atol=1e-9) else angle


def amplification(matrix: np.ndarray, directions: np.ndarray) -> np.ndarray:
    """|M v| for each row v of `directions`.

    With M = Phi(tau) and unit vectors v this is the factor by which a
    perturbation along v grows (> 1) or shrinks (< 1) over a time tau.
    """
    return np.linalg.norm(directions @ matrix.T, axis=1)


def leading_singular_direction(matrix: np.ndarray) -> tuple[float, np.ndarray]:
    """Largest singular value sigma_1 of a 2x2 matrix M and its right singular vector s_1.

    s_1 is the unit vector v that maximises |M v|, and sigma_1 is that maximum.
    For M = Phi(tau), s_1 is the perturbation that grows the most over the
    horizon tau (the leading finite-time singular vector).
    """
    _, singular_values, vt = np.linalg.svd(matrix)
    return float(singular_values[0]), _canonical(vt[0])


def lyapunov_directions_2d(A: np.ndarray) -> dict[str, np.ndarray]:
    """Named unit directions of a constant 2x2 matrix A with real eigenvalues l1 > l2.

    For a linear system with constant A the standard objects of Lyapunov
    analysis have closed forms:

    clv1, clv2  Eigenvectors of A for l1 and l2. These are the covariant
                Lyapunov vectors: the dynamics maps each onto a multiple of itself.
    b1, b2      Backward (Gram-Schmidt) Lyapunov vectors: what "propagate an
                orthonormal basis forward and re-orthonormalise it with QR"
                converges to. Long propagation turns almost every vector toward
                clv1, so b1 = clv1, and b2 is perpendicular to b1.
    f1, f2      Forward Lyapunov vectors: right singular vectors of Phi(tau)
                as tau grows. Every direction except clv2 eventually grows at
                rate l1, so the least-growing direction is f2 = clv2, and f1 is
                perpendicular to clv2.

    If A is symmetric all three pairs coincide; if A is non-normal they differ.
    Each vector is returned with unit length and angle in [0, 180) degrees.
    """
    eigenvalues, eigenvectors = np.linalg.eig(A)
    # Recent NumPy returns complex arrays even when every eigenvalue is real.
    if np.any(np.abs(np.imag(eigenvalues)) > 1e-12):
        raise ValueError("A must have real eigenvalues")
    eigenvalues, eigenvectors = np.real(eigenvalues), np.real(eigenvectors)
    order = np.argsort(eigenvalues)[::-1]
    l1, l2 = eigenvalues[order]
    if np.isclose(l1, l2):
        raise ValueError("eigenvalues must be distinct; with equal rates no direction is special")
    clv1 = _canonical(eigenvectors[:, order[0]])
    clv2 = _canonical(eigenvectors[:, order[1]])
    return {
        "clv1": clv1,
        "clv2": clv2,
        "b1": clv1,
        "b2": _canonical(_rotate_90(clv1)),
        "f1": _canonical(_rotate_90(clv2)),
        "f2": clv2,
    }


def _rotate_90(v: np.ndarray) -> np.ndarray:
    return np.array([-v[1], v[0]])


def _canonical(v: np.ndarray) -> np.ndarray:
    """Unit vector along v with its angle mapped into [0, 180) degrees."""
    return unit_vectors(np.radians(direction_angle(v)))[0]
