"""Write the job files of each gate (docs/plan.md, sections 7-10).

    python experiments/make_jobs.py gate0    # jobs/gate0/*.json
    python experiments/make_jobs.py gate1    # also freezes results/gate1/initial_conditions.json
    python experiments/make_jobs.py gate2
    python experiments/make_jobs.py side
    python experiments/make_jobs.py pilot    # timing pilot, window 0 / one attractor state
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mw.reference import damped_initial_state, lorenz_attractor_states, reference  # noqa: E402
from mw.systems import DavisSkodje, LinearSaddle, Lorenz63, System  # noqa: E402
from mw.train import Settings, grid  # noqa: E402

ALL6 = ["U", "TUR", "S-orc", "M-orc", "S-self", "M-self"]
LINEAR4 = ["U", "TUR", "S-orc", "M-orc"]  # self = oracle when the Jacobian is constant
SEEDS = [0, 1, 2]
IC_FILE = ROOT / "results" / "gate1" / "initial_conditions.json"


def dial_scaling(system: System, T: float) -> tuple[list[float], list[float]]:
    """Midrange and half-range of the reference over dial windows 0-8 on [0, T]."""
    t = grid(T, 2001)
    xs = np.concatenate([reference(system, system.initial_state(w), t) for w in range(9)])
    lo, hi = xs.min(axis=0), xs.max(axis=0)
    half = np.maximum(0.5 * (hi - lo), 1e-3)
    return (0.5 * (hi + lo)).tolist(), half.tolist()


def lorenz_scaling(k: int) -> tuple[list[float], list[float]]:
    return [0.0, 0.0, 25.0] + [0.0] * k, [10.0] * (3 + k)


def frozen_lorenz_states() -> list[dict]:
    if IC_FILE.exists():
        return json.loads(IC_FILE.read_text())["states"]
    states = lorenz_attractor_states(12)
    IC_FILE.parent.mkdir(parents=True, exist_ok=True)
    record = {"description": "Gate 1 attractor states (docs/plan.md, section 8): spin-up 100 from (1,1,1), "
                             "12 sorted uniform times in [0, 600], NumPy seed 2026112201",
              "states": [{"id": i + 1, "x0": s.tolist()} for i, s in enumerate(states)]}
    IC_FILE.write_text(json.dumps(record, indent=2) + "\n")
    return record["states"]


def write_job(directory: Path, name: str, job: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{name}.json").write_text(json.dumps(job, indent=1) + "\n")


def job(system: System, T: float, method: str, ics: list[dict], seeds: list[int], out_dir: str,
        mean, scale, threads: int = 1, **settings) -> dict:
    s = Settings(t_final=T, **settings)
    return {"system": system.to_dict(), "method": method, "settings": s.__dict__, "ics": ics, "seeds": seeds,
            "out_dir": out_dir, "out_mean": mean, "out_scale": scale, "threads": threads}


def gate0(out: Path) -> None:
    T = 2.0
    for system, methods in [*[(LinearSaddle(k), LINEAR4) for k in (1, 4, 16, 64)],
                            *[(DavisSkodje(g), ALL6) for g in (3, 6, 18, 66)]]:
        mean, scale = dial_scaling(system, T)
        ics = [{"id": w, "x0": system.initial_state(w).tolist()} for w in range(1, 9)]
        for m in methods:
            write_job(out, f"{system.name}_T2_{m}",
                      job(system, T, m, ics, SEEDS, f"results/gate0/{system.name}/T2/{m}", mean, scale))


def gate1(out: Path) -> None:
    ics = frozen_lorenz_states()
    mean, scale = lorenz_scaling(0)
    for T in (0.5, 1.0, 2.0):
        for m in ALL6:
            write_job(out, f"lorenz63_T{T:g}_{m}",
                      job(Lorenz63(), T, m, ics, SEEDS, f"results/gate1/lorenz63/T{T:g}/{m}", mean, scale))


def gate2(out: Path) -> None:
    states = frozen_lorenz_states()
    for k in (0, 3, 9, 29):
        system = Lorenz63(k)
        mean, scale = lorenz_scaling(k)
        ics = [{"id": s["id"], "x0": damped_initial_state(np.array(s["x0"]), k).tolist()} for s in states]
        for m in ALL6:
            write_job(out, f"{system.name}_T1_{m}",
                      job(system, 1.0, m, ics, SEEDS, f"results/gate2/{system.name}/T1/{m}", mean, scale))


def side(out: Path) -> None:
    ics = frozen_lorenz_states()
    mean, scale = lorenz_scaling(0)
    L = Lorenz63()
    for T in (0.5, 1.0, 2.0):
        for m in ("M-final-orc", "M-final-self"):
            write_job(out, f"final_T{T:g}_{m}", job(L, T, m, ics, SEEDS, f"results/side/final/T{T:g}/{m}", mean, scale))
        for eps in (0.1, 1.0, 10.0):
            write_job(out, f"causal_T{T:g}_eps{eps:g}",
                      job(L, T, "causal", ics, SEEDS, f"results/side/causal/T{T:g}/eps{eps:g}", mean, scale,
                          causal_eps=eps))
    for tag, extra in [("iters50k", {"iterations": 50000}), ("width128", {"hidden": [128] * 4}),
                       ("grid1024", {"n_grid": 1024})]:
        for m in ("U", "TUR", "S-self", "M-self"):
            write_job(out, f"sens_{tag}_T1_{m}",
                      job(L, 1.0, m, ics, SEEDS, f"results/side/sensitivity/{tag}/T1/{m}", mean, scale, **extra))
    turinici_ic = [{"id": 0, "x0": [1.0, 1.0, 1.0]}]
    for m in ALL6:
        write_job(out, f"turinici_bench_{m}",
                  job(L, 0.5, m, turinici_ic, list(range(10)), f"results/side/turinici_benchmark/T0.5/{m}", mean,
                      scale, hidden=[20] * 5, lr=0.01, lr_final=0.01, iterations=2000, n_grid=256, log_every=100))


def pilot(out: Path) -> None:
    """Timing pilot: one batch per family with the full step count (results kept separate)."""
    sad = LinearSaddle(64.0)
    mean, scale = dial_scaling(sad, 2.0)
    ics0 = [{"id": 0, "x0": sad.initial_state(0).tolist()}]
    for m in ("U", "M-orc"):
        write_job(out, f"pilot_saddle_{m}", job(sad, 2.0, m, ics0, SEEDS, f"results/pilot/saddle_k64/{m}", mean, scale))
    L = Lorenz63()
    lics = frozen_lorenz_states()[:1]
    lm, ls = lorenz_scaling(0)
    for m in ("U", "M-self"):
        write_job(out, f"pilot_lorenz_T2_{m}", job(L, 2.0, m, lics, SEEDS, f"results/pilot/lorenz63_T2/{m}", lm, ls))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("gate", choices=["gate0", "gate1", "gate2", "side", "pilot"])
    args = ap.parse_args()
    out = ROOT / "jobs" / args.gate
    {"gate0": gate0, "gate1": gate1, "gate2": gate2, "side": side, "pilot": pilot}[args.gate](out)
    print(f"wrote {len(list(out.glob('*.json')))} job files to {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
