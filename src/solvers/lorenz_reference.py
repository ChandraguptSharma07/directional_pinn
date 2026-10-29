"""High-accuracy Lorenz-63 reference windows (Phase 2).

Design (explained in docs/phase2_lorenz_reference.md):

1. A long run starts from ``seed_state`` and discards ``transient`` time units,
   so that everything recorded afterwards lies on the attractor.
2. Window k starts at time t_k = transient + spinup + k * gap of the long run.
   Its initial condition x0_k is the long run's state at t_k, and the stretch
   [t_k - spinup, t_k] of the long run is the window's past. The backward
   Lyapunov vector b1 needs that past (Phase 3); Lorenz cannot be integrated
   backward in time without enormous error growth. Windows are ``gap`` time
   units apart, many Lyapunov times, so they behave like independent initial
   conditions.
3. The reference for window k is a fresh integration from x0_k, so that its
   accuracy depends only on errors made inside the window.

All times inside a window are measured from its start: the window covers
[0, length] and its past covers [-spinup, 0].
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.solvers.reference_solver import ODESolution, integrate
from src.systems.lorenz63 import LorenzParams, lorenz_rhs


@dataclass(frozen=True, eq=False)
class ReferenceWindow:
    """One initial condition with its high-accuracy reference and its past."""

    index: int
    start_in_long_run: float  # t_k
    spinup: float
    x0: np.ndarray
    reference: ODESolution  # fresh integration from x0, with dense output
    long_run: ODESolution  # the long run, with dense output

    def past(self, t) -> np.ndarray:
        """States at window times t in [-spinup, 0], taken from the long run."""
        t = np.asarray(t, dtype=float)
        if np.any(t < -self.spinup - 1e-12) or np.any(t > 1e-12):
            raise ValueError("past() is defined on [-spinup, 0]")
        return self.long_run.at(self.start_in_long_run + t)


def window_start_times(transient: float, spinup: float, gap: float, n_windows: int) -> np.ndarray:
    """Start times t_k of the windows along the long run."""
    return transient + spinup + gap * np.arange(n_windows)


def build_reference_windows(
    params: LorenzParams,
    seed_state,
    transient: float,
    spinup: float,
    gap: float,
    n_windows: int,
    length: float,
    dt: float,
    solver: dict,
) -> list[ReferenceWindow]:
    """Long run, initial conditions, and one fresh reference integration per window.

    The reference of each window is sampled every `dt` on [0, length] and has
    dense output. `solver` holds the method and tolerances (used for both the
    long run and the references).
    """
    rhs = lambda t, state: lorenz_rhs(t, state, params)
    starts = window_start_times(transient, spinup, gap, n_windows)
    long_run = integrate(rhs, seed_state, np.concatenate([[0.0], starts]), dense_output=True, **solver)
    t_grid = np.linspace(0.0, length, int(round(length / dt)) + 1)
    windows = []
    for k, start in enumerate(starts):
        x0 = long_run.y[k + 1].copy()
        reference = integrate(rhs, x0, t_grid, dense_output=True, **solver)
        windows.append(
            ReferenceWindow(
                index=k, start_in_long_run=float(start), spinup=spinup, x0=x0, reference=reference, long_run=long_run
            )
        )
    return windows


def build_windows_from_config(params: LorenzParams, cfg: dict) -> list[ReferenceWindow]:
    """``build_reference_windows`` with settings taken from the config section ``lorenz_reference``."""
    return build_reference_windows(
        params,
        seed_state=np.asarray(cfg["seed_state"], dtype=float),
        transient=float(cfg["transient"]),
        spinup=float(cfg["spinup"]),
        gap=float(cfg["gap"]),
        n_windows=int(cfg["n_windows"]),
        length=float(cfg["length"]),
        dt=float(cfg["dt"]),
        solver=solver_settings(cfg["solver"]),
    )


def solver_settings(cfg: dict) -> dict:
    """Solver method and tolerances from a config mapping, with explicit types."""
    return {"method": str(cfg["method"]), "rtol": float(cfg["rtol"]), "atol": float(cfg["atol"])}
