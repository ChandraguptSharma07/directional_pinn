"""Planned analysis of every gate (docs/plan.md, sections 6-9).

Reads results/<gate>/**/summary.json and writes results/<gate>/analysis.json
and analysis.md. Contrasts pair networks by (initial condition, seed), average
over seeds within an initial condition, and resample initial conditions.

    python experiments/analyze.py gate0 gate1 gate2 side
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CAP = 1e3  # errors above this, or non-finite, are set to it (plan, section 6)
BEAT = -np.log10(1.5)  # at least 1.5 times lower error
N_BOOT = 10_000


def load(gate: str, lorenz_part: bool = False) -> dict:
    """{(system, T, method): {(ic, seed): metrics}} for one gate."""
    runs: dict = defaultdict(dict)
    for f in sorted((ROOT / "results" / gate).rglob("summary.json")):
        s = json.loads(f.read_text())
        job = s["job"]
        rel = Path(job["out_dir"]).parts  # results/<gate>/<system>/<T..>/<method..>
        system, T, method = rel[2], rel[3], "/".join(rel[4:])
        m = s.get("metrics_lorenz_part") if lorenz_part and "metrics_lorenz_part" in s else s["metrics"]
        for i, p in enumerate(s["pairs"]):
            runs[(system, T, method)][(p["ic"], p["seed"])] = {
                "E_rms": m["E_rms"][i], "E_max": m["E_max"][i], "E_T": m["E_T"][i],
                "success": m["success_max_below_1"][i], "runtime": s["runtime_seconds"],
                "device": s.get("device"), "host": s.get("host"),
            }
    return runs


def log_err(x) -> float:
    x = float(x)
    if not np.isfinite(x) or x > CAP:
        x = CAP
    return float(np.log10(max(x, 1e-12)))


def contrast(runs: dict, key_a: tuple, key_b: tuple, metric: str = "E_rms") -> dict | None:
    a, b = runs.get(key_a), runs.get(key_b)
    if not a or not b:
        return None
    common = sorted(set(a) & set(b))
    by_ic = defaultdict(list)
    for ic, seed in common:
        by_ic[ic].append(log_err(a[(ic, seed)][metric]) - log_err(b[(ic, seed)][metric]))
    per_ic = np.array([np.mean(v) for _, v in sorted(by_ic.items())])
    rng = np.random.default_rng(0)
    boots = per_ic[rng.integers(0, len(per_ic), size=(N_BOOT, len(per_ic)))].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    mean = float(per_ic.mean())
    pairs = np.array([log_err(a[k][metric]) - log_err(b[k][metric]) for k in common])
    return {"A": key_a[2], "B": key_b[2], "n_pairs": len(common), "n_ics": len(per_ic), "mean_delta": mean,
            "ci95": [float(lo), float(hi)], "fold_change": float(10 ** mean),
            "fraction_pairs_A_lower": float(np.mean(pairs < 0)),
            "A_beats_B": bool(mean <= BEAT and hi < 0), "A_worse_than_B": bool(lo > 0),
            "per_ic_delta": per_ic.tolist()}


def describe(runs: dict) -> list[dict]:
    rows = []
    for (system, T, method), d in sorted(runs.items()):
        e = np.array([log_err(v["E_rms"]) for v in d.values()])
        emax = np.array([log_err(v["E_max"]) for v in d.values()])
        rows.append({"system": system, "T": T, "method": method, "n": len(d),
                     "median_log10_E_rms": float(np.median(e)), "q25": float(np.percentile(e, 25)),
                     "q75": float(np.percentile(e, 75)), "median_log10_E_max": float(np.median(emax)),
                     "success_max_below_1": int(sum(v["success"] for v in d.values())),
                     "median_runtime_s": float(np.median([v["runtime"] for v in d.values()])),
                     "devices": sorted({str(v["device"]) for v in d.values()})})
    return rows


def fmt_contrast(c: dict | None) -> str:
    if c is None:
        return "n/a"
    verdict = "**beats**" if c["A_beats_B"] else ("worse" if c["A_worse_than_B"] else "no decision")
    return (f"{c['A']} vs {c['B']}: mean Δ {c['mean_delta']:+.3f} [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}] "
            f"(×{c['fold_change']:.2f}; A lower in {100 * c['fraction_pairs_A_lower']:.0f}% of pairs) → {verdict}")


def table_md(rows: list[dict]) -> str:
    out = ["| system | T | method | n | median log10 E_rms [IQR] | median log10 E_max | E_max<1 |",
           "|---|---|---|---:|---|---:|---:|"]
    for r in rows:
        out.append(f"| {r['system']} | {r['T']} | {r['method']} | {r['n']} | {r['median_log10_E_rms']:.2f} "
                   f"[{r['q25']:.2f}, {r['q75']:.2f}] | {r['median_log10_E_max']:.2f} | "
                   f"{r['success_max_below_1']}/{r['n']} |")
    return "\n".join(out)


def gate0() -> dict:
    runs = load("gate0")
    res = {"describe": describe(runs), "families": {}}
    lines = ["# Gate 0 analysis (planned rules, docs/plan.md section 7)", ""]
    fams = {"saddle": [("saddle_k1", 2), ("saddle_k4", 5), ("saddle_k16", 17), ("saddle_k64", 65)],
            "davis_skodje": [("davis_skodje_g3", 2), ("davis_skodje_g6", 5), ("davis_skodje_g18", 17),
                             ("davis_skodje_g66", 65)]}
    for fam, settings in fams.items():
        out = []
        for system, dial in settings:
            k = lambda m: (system, "T2", m)  # noqa: E731
            c = {"M_vs_S": contrast(runs, k("M-orc"), k("S-orc")), "M_vs_U": contrast(runs, k("M-orc"), k("U")),
                 "M_vs_TUR": contrast(runs, k("M-orc"), k("TUR")), "S_vs_U": contrast(runs, k("S-orc"), k("U"))}
            if fam == "davis_skodje":
                c["Mself_vs_Morc"] = contrast(runs, k("M-self"), k("M-orc"))
                c["Mself_vs_Sself"] = contrast(runs, k("M-self"), k("S-self"))
            out.append({"system": system, "dial": dial, **c})
        complete = all(o["M_vs_S"] is not None for o in out)
        beats = [bool(o["M_vs_S"] and o["M_vs_S"]["A_beats_B"]) for o in out]
        gain = [-(o["M_vs_S"]["mean_delta"]) if o["M_vs_S"] else None for o in out]
        supported = complete and beats[2] and beats[3] and gain[3] > gain[0]
        res["families"][fam] = {"settings": out, "complete": complete, "M_beats_S": beats, "gain": gain,
                                "supported_in_family": supported}
        lines += [f"## {fam}", ""]
        for o in out:
            lines.append(f"- **{o['system']} (dial {o['dial']})**")
            for name in [n for n in o if n not in ("system", "dial")]:
                lines.append(f"  - {fmt_contrast(o[name])}")
        lines += ["", f"Family complete: {complete}; M beats S at dial settings: {beats}; supported in family: {supported}", ""]
    fam_res = res["families"].values()
    if not all(f["complete"] for f in fam_res):
        verdict = "incomplete"
    elif any(f["supported_in_family"] for f in fam_res):
        verdict = "supported"
    elif not any(any(f["M_beats_S"]) for f in fam_res):
        verdict = "rejected"
    else:
        verdict = "partial"
    res["H1"] = verdict
    lines += [f"**H1: {verdict}**", "", "## Descriptive table", "", table_md(res["describe"])]
    write("gate0", res, lines)
    return res


def gate1() -> dict:
    runs = load("gate1")
    res = {"describe": describe(runs), "horizons": {}}
    lines = ["# Gate 1 analysis (planned rules, docs/plan.md section 8)", ""]
    beats_any = {m: False for m in ("S-self", "U", "TUR")}
    worse_any = False
    complete = True
    for T in ("T0.5", "T1", "T2"):
        k = lambda m: ("lorenz63", T, m)  # noqa: E731
        c = {f"Mself_vs_{m}": contrast(runs, k("M-self"), k(m)) for m in ("S-self", "U", "TUR")}
        c["Morc_vs_Sorc"] = contrast(runs, k("M-orc"), k("S-orc"))
        c["Sself_vs_U"] = contrast(runs, k("S-self"), k("U"))
        c["TUR_vs_U"] = contrast(runs, k("TUR"), k("U"))
        for m in ("S-self", "U", "TUR"):
            cm = c[f"Mself_vs_{m}"]
            if cm is None:
                complete = False
                continue
            beats_any[m] |= cm["A_beats_B"]
            worse_any |= cm["A_worse_than_B"]
        res["horizons"][T] = c
        lines += [f"## {T}", ""] + [f"- {fmt_contrast(v)}" for v in c.values()] + [""]
    if not complete:
        verdict = "incomplete"
    elif all(beats_any.values()) and not worse_any:
        verdict = "supported"
    elif not any(beats_any.values()):
        verdict = "rejected"
    else:
        verdict = "partial"
    res["H2"] = verdict
    res["beats_any_horizon"] = beats_any
    res["worse_at_some_horizon"] = worse_any
    lines += [f"**H2: {verdict}** (beats at some horizon: {beats_any}; worse somewhere: {worse_any})", "",
              "## Descriptive table", "", table_md(res["describe"])]
    write("gate1", res, lines)
    return res


def gate2() -> dict:
    runs = load("gate2", lorenz_part=True)
    res = {"describe": describe(runs), "by_k": {}}
    lines = ["# Gate 2 analysis (planned rules, docs/plan.md section 9; Lorenz-part error)", ""]
    names = {0: "lorenz63", 3: "lorenz63_damped3", 9: "lorenz63_damped9", 29: "lorenz63_damped29"}
    per_ic_gain = {}
    for kk, system in names.items():
        k = lambda m: (system, "T1", m)  # noqa: E731
        c = {"Mself_vs_U": contrast(runs, k("M-self"), k("U")), "Mself_vs_Sself": contrast(runs, k("M-self"), k("S-self")),
             "Mself_vs_TUR": contrast(runs, k("M-self"), k("TUR")), "Morc_vs_Sorc": contrast(runs, k("M-orc"), k("S-orc"))}
        res["by_k"][kk] = c
        if c["Mself_vs_U"]:
            per_ic_gain[kk] = -np.array(c["Mself_vs_U"]["per_ic_delta"])
        lines += [f"## k = {kk}", ""] + [f"- {fmt_contrast(v)}" for v in c.values()] + [""]
    if 0 in per_ic_gain and 29 in per_ic_gain and len(per_ic_gain[0]) == len(per_ic_gain[29]):
        diff = per_ic_gain[29] - per_ic_gain[0]
        rng = np.random.default_rng(0)
        boots = diff[rng.integers(0, len(diff), size=(N_BOOT, len(diff)))].mean(axis=1)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        res["gain_difference_k29_minus_k0"] = {"mean": float(diff.mean()), "ci95": [float(lo), float(hi)]}
        res["H3"] = "supported" if lo > 0 else "not supported"
        lines.append(f"**H3: {res['H3']}** (gain at k=29 minus gain at k=0: {diff.mean():+.3f} [{lo:+.3f}, {hi:+.3f}])")
    else:
        res["H3"] = "incomplete"
        lines.append("**H3: incomplete**")
    lines += ["", "## Descriptive table (Lorenz-part error)", "", table_md(res["describe"])]
    write("gate2", res, lines)
    return res


def side() -> dict:
    runs = load("side")
    res = {"describe": describe(runs)}
    write("side", res, ["# Side runs (exploratory)", "", table_md(res["describe"])])
    return res


def write(gate: str, res: dict, lines: list[str]) -> None:
    d = ROOT / "results" / gate
    d.mkdir(parents=True, exist_ok=True)
    (d / "analysis.json").write_text(json.dumps(res, indent=1) + "\n")
    (d / "analysis.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:60]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("gates", nargs="+", choices=["gate0", "gate1", "gate2", "side"])
    args = ap.parse_args()
    for g in args.gates:
        {"gate0": gate0, "gate1": gate1, "gate2": gate2, "side": side}[g]()


if __name__ == "__main__":
    sys.exit(main())
