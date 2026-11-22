"""Mechanism check (exploratory): where does each method put its leftover residual?

For every trained network, along the reference path on the evaluation grid:

    plain    = sqrt( mean_t |r(t)|^2 )                       size of the residual
    harm     = mean_t sqrt( r(t)^T G(t) r(t) ) * T            the window-error bound (theory.md, eq. 2)
    harm_per_size = harm / (plain * T)                         typical harm per unit residual
    top_frac = mean_t (r . g1)^2 / mean_t |r|^2                residual energy along the most harmful direction g1 of G(t)
    low_frac = same along the least harmful direction

G(t) is the oracle window Gramian (reference Jacobians), so this is a
diagnostic, not part of any method. The theory predicts that the matrix weight
moves the residual away from harmful directions: lower harm_per_size and
top_frac, and higher low_frac, than uniform weighting.

    python experiments/mechanism.py gate1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mw import weights as wt  # noqa: E402
from mw.systems import system_from_dict  # noqa: E402


def analyse_run(summary_file: Path) -> dict | None:
    data_file = summary_file.with_name("data.npz")
    if not data_file.exists():
        return None
    s = json.loads(summary_file.read_text())
    system = system_from_dict(s["job"]["system"])
    d = np.load(data_file)
    r, ref, t = d["r_eval"], d["ref_eval"], d["t_eval"]
    h = float(t[1] - t[0])
    T = float(t[-1])
    x = torch.tensor(ref, dtype=torch.float64)
    G = wt.window_gramian(wt.step_propagators(system.jac(x), h), h).numpy()  # (B, Ne, n, n)
    evals, evecs = np.linalg.eigh(G)  # ascending
    rr = np.einsum("bti,btij,btj->bt", r, G, r)
    plain = np.sqrt(np.mean(np.sum(r**2, -1), axis=1))
    harm = np.mean(np.sqrt(np.clip(rr, 0, None)), axis=1) * T
    e2 = np.sum(r**2, -1).mean(axis=1)
    top = np.mean(np.einsum("bti,bti->bt", r, evecs[..., :, -1]) ** 2, axis=1) / e2
    low = np.mean(np.einsum("bti,bti->bt", r, evecs[..., :, 0]) ** 2, axis=1) / e2
    rel = Path(s["job"]["out_dir"]).parts
    return {"system": rel[2], "T": rel[3], "method": "/".join(rel[4:]), "plain": plain.tolist(),
            "harm": harm.tolist(), "harm_per_size": (harm / (plain * T)).tolist(), "top_frac": top.tolist(),
            "low_frac": low.tolist(), "E_rms": s["metrics"]["E_rms"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("gate")
    args = ap.parse_args()
    rows = [r for f in sorted((ROOT / "results" / args.gate).rglob("summary.json")) if (r := analyse_run(f))]
    out = ROOT / "results" / args.gate / "mechanism.json"
    out.write_text(json.dumps(rows) + "\n")
    print("| system | T | method | median plain residual | median harm bound | median harm per unit residual "
          "| median top-direction share | median low-direction share | median E_rms |")
    print("|---|---|---|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        med = {k: float(np.median(r[k])) for k in ("plain", "harm", "harm_per_size", "top_frac", "low_frac", "E_rms")}
        print(f"| {r['system']} | {r['T']} | {r['method']} | {med['plain']:.3g} | {med['harm']:.3g} | "
              f"{med['harm_per_size']:.3g} | {med['top_frac']:.3f} | {med['low_frac']:.3f} | {med['E_rms']:.3g} |")


if __name__ == "__main__":
    main()
