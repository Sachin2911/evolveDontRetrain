"""Stop rule at generation 8 (graded/PREREG.md) and learning diagnostics for any generation count.

  python -m graded.stoprule <out_dir>/<adapter>-r<k>/state.json [--gens 8] [--nulls 300]

Measures (CMA-ES vs random search, same generations, same cases):
  trend   Spearman rho of the scalar S*100 + C*10 - depth over every sample against its generation, per arm
  moved   distance of CMA-ES's mean from the start (box centre) vs the coin-flip null's 95th percentile
          (fresh CMA-ES with the same options fed random ranks)
"""

import argparse
import json
from pathlib import Path

import cma
import numpy as np
from scipy.stats import spearmanr


def scalar(c):
    return c["S"] * 100 + c["C"] * 10 - c["depth"]


def trend(gens, arm):
    gi, sc = [], []
    for g in gens:
        for c in g[arm]:
            gi.append(g["generation"])
            sc.append(scalar(c))
    if len(set(sc)) < 2:
        return 0.0, 1.0
    r = spearmanr(gi, sc)
    return float(r.statistic), float(r.pvalue)


def opts(pop, seed):
    return {"popsize": pop, "CMA_diagonal": True, "bounds": [-1.0, 1.0], "seed": seed, "verbose": -9, "verb_log": 0,
            "verb_disp": 0, "tolfun": 0, "tolfunhist": 0, "tolflatfitness": 10**6, "tolx": 0, "maxiter": 10**6}


def moved_real(gens, n, pop, sigma0):
    es = cma.CMAEvolutionStrategy(np.zeros(n), sigma0, opts(pop, 1))
    for g in gens:
        es.ask()
        es.tell([np.asarray(x) for x in g["cma_x"]], g["cma_ranks"])
    return float(np.linalg.norm(es.mean))


def moved_null(n_gen, n, pop, sigma0, seed):
    rng = np.random.default_rng(seed)
    es = cma.CMAEvolutionStrategy(np.zeros(n), sigma0, opts(pop, seed + 1))
    for _ in range(n_gen):
        x = es.ask()
        es.tell(x, list(rng.permutation(pop)))
    return float(np.linalg.norm(es.mean))


def check(state_path, n_gens=None, nulls=300, sigma0=0.6):
    s = json.loads(Path(state_path).read_text())
    gens = s["generations"][: n_gens] if n_gens else s["generations"]
    n = len(gens[0]["cma_x"][0])
    pop = len(gens[0]["cma_x"])
    rc, pc = trend(gens, "cma")
    rr, pr = trend(gens, "rnd")
    m = moved_real(gens, n, pop, sigma0)
    null = np.array([moved_null(len(gens), n, pop, sigma0, i) for i in range(nulls)])
    q95 = float(np.quantile(null, 0.95))
    first = lambda arm: float(np.mean([scalar(c) for c in gens[0][arm]]))
    last = lambda arm: float(np.mean([scalar(c) for c in gens[-1][arm]]))
    out = {
        "state": str(state_path), "generations": len(gens), "dim": n, "pop": pop,
        "trend_cma": [rc, pc], "trend_rnd": [rr, pr], "trend_pass": bool(rc > rr and pc < 0.05),
        "moved": m, "null_q95": q95, "moved_pass": bool(m > q95),
        "first_last_cma": [first("cma"), last("cma")], "first_last_rnd": [first("rnd"), last("rnd")],
        "share_rnd_beating_native": float(np.mean([tuple(c["improvement"]) > (0,) * len(c["improvement"])
                                                   for g in gens for c in g["rnd"]])),
        # NOTE: assumes 9 cases per candidate (run 1); run-2 searches used 12, so run-2 values are 4/3 too high.
        "free_share": float(np.mean([c.get("free", 0) / 9.0 for g in gens for arm in ("cma", "rnd") for c in g[arm]])),
    }
    out["pass"] = out["trend_pass"] or out["moved_pass"]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("state", nargs="+")
    ap.add_argument("--gens", type=int, default=None)
    ap.add_argument("--nulls", type=int, default=300)
    args = ap.parse_args()
    for p in args.state:
        print(json.dumps(check(p, args.gens, args.nulls)), flush=True)


if __name__ == "__main__":
    main()
