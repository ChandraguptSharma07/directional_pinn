"""Numerical integration of ODEs with scipy's solve_ivp, keeping a record of the settings.

Every result file in this project should state exactly how its reference
solution was computed, so ``integrate`` returns the solver settings and work
statistics together with the solution.
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
    """

    t: np.ndarray
    y: np.ndarray
    settings: dict
    stats: dict


def integrate(
    rhs: Callable[[float, np.ndarray], np.ndarray],
    y0,
    t_eval,
    method: str = "DOP853",
    rtol: float = 1e-10,
    atol: float = 1e-12,
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
    sol = solve_ivp(rhs, (t_eval[0], t_eval[-1]), y0, method=method, t_eval=t_eval, rtol=rtol, atol=atol)
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
    return ODESolution(t=sol.t, y=sol.y.T, settings=settings, stats=stats)
