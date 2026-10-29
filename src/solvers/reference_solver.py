"""Numerical integration of ODEs with scipy's solve_ivp, keeping a record of the settings.

Every result file in this project should state exactly how its reference
solution was computed, so ``integrate`` returns the solver settings and work
statistics together with the solution. ``tolerance_study`` re-solves a problem
with several settings to estimate how far a solution can be trusted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.integrate import solve_ivp


@dataclass
class ODESolution:
    """A numerical solution sampled at the requested times.

    Attributes:
        t: sample times, shape (N,).
        y: states, shape (N, d); row i is the state at t[i]. This is the
            transpose of solve_ivp's (d, N) layout, so that rows are time samples
            as everywhere else in the project.
        settings: integration method and tolerances.
        stats: work counters and status reported by solve_ivp.
        interpolant: solve_ivp's dense-output interpolant (only if requested).
    """

    t: np.ndarray
    y: np.ndarray
    settings: dict
    stats: dict
    interpolant: Callable | None = None

    def at(self, t) -> np.ndarray:
        """States at arbitrary times inside the integration interval, shape (..., d).

        Uses the method's own interpolant, whose error is of the same order as
        the integration error. Requires ``dense_output=True`` in ``integrate``.
        """
        if self.interpolant is None:
            raise ValueError("this solution was computed without dense output")
        t = np.asarray(t, dtype=float)
        return np.moveaxis(self.interpolant(t), 0, -1)


def integrate(
    rhs: Callable[[float, np.ndarray], np.ndarray],
    y0,
    t_eval,
    method: str = "DOP853",
    rtol: float = 1e-10,
    atol: float = 1e-12,
    dense_output: bool = False,
) -> ODESolution:
    """Integrate dy/dt = rhs(t, y) from y(t_eval[0]) = y0 and sample the solution at t_eval.

    DOP853 is an explicit Runge-Kutta method of order 8 with adaptive step size:
    each step is accepted only if the estimated local error in every component
    is below atol + rtol * |y|. The global error at the end can still exceed
    rtol, because local errors accumulate (and grow in unstable directions).

    Raises:
        RuntimeError: if the solver does not reach the final time.
    """
    t_eval = np.asarray(t_eval, dtype=float)
    y0 = np.asarray(y0, dtype=float)
    sol = solve_ivp(
        rhs, (t_eval[0], t_eval[-1]), y0, method=method, t_eval=t_eval, rtol=rtol, atol=atol, dense_output=dense_output
    )
    if not sol.success:
        raise RuntimeError(f"solve_ivp ({method}) failed: {sol.message}")
    settings = {
        "method": method,
        "rtol": rtol,
        "atol": atol,
        "t_span": [float(t_eval[0]), float(t_eval[-1])],
        "n_eval": int(t_eval.size),
    }
    stats = {
        "nfev": int(sol.nfev),
        "njev": int(sol.njev),
        "nlu": int(sol.nlu),
        "status": int(sol.status),
        "message": sol.message,
    }
    return ODESolution(t=sol.t, y=sol.y.T, settings=settings, stats=stats, interpolant=sol.sol)


def setting_label(setting: dict) -> str:
    """Short name such as 'DOP853 rtol=1e-10' for a solver setting."""
    return f"{setting['method']} rtol={setting['rtol']:.0e}"


def tolerance_study(
    rhs: Callable[[float, np.ndarray], np.ndarray],
    y0,
    t_eval,
    settings: list[dict],
    strictest: dict,
) -> tuple[ODESolution, dict[str, np.ndarray]]:
    """Solve the same problem with every setting and compare each with the strictest one.

    Returns the strictest solution and, for every setting, the distance
    |y_setting(t) - y_strictest(t)| (Euclidean norm) at each time in t_eval.
    If the strictest solution is much more accurate than the others, that
    distance estimates each setting's own error.
    """
    best = integrate(rhs, y0, t_eval, **strictest)
    deviations = {
        setting_label(setting): np.linalg.norm(integrate(rhs, y0, t_eval, **setting).y - best.y, axis=1)
        for setting in settings
    }
    return best, deviations


def trusted_until(t: np.ndarray, deviation: np.ndarray, threshold: float) -> float:
    """Last time before `deviation` first exceeds `threshold` (t[-1] if it never does)."""
    exceeded = np.nonzero(deviation > threshold)[0]
    if exceeded.size == 0:
        return float(t[-1])
    return float(t[exceeded[0] - 1]) if exceeded[0] > 0 else float(t[0])
