"""Summary numbers for the v2-only paper text (ranges and counts over the locked run-2 adapters).

  python paper/make_summary.py

Writes summary.tex (\\newcommand per number, prefix Sum...). Reads the same evidence as make_numbers.py through
figures/gdata.py (read-only). Kept separate from make_numbers.py so the two never edit the same file.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "figures"))
import gdata as G  # noqa: E402

M = {}


def put(k, v):
    assert k.isalpha(), k
    M[k] = v


def main():
    pc = G.per_case()
    locked = [a for a in G.arms(2) if G.arm_info(a)["method"] in ("cma", "rnd")]
    nat = pc["native"]
    put("SumLockedN", len(locked))
    tr = {a: G.counts(a, G.TRAIN) for a in locked}
    rc = {a: G.counts(a, ("reacher",)) for a in locked}
    put("SumTrainSMin", min(v["S"] for v in tr.values()))
    put("SumTrainSMax", max(v["S"] for v in tr.values()))
    sig = [a for a in locked if (lambda p: p["p"] < 0.05 and p["gain"] > p["loss"])(G.paired(a, "native", G.TRAIN))]
    put("SumSigTrainN", len(sig))
    sig_r = [a for a in locked if (lambda p: p["p"] < 0.05 and p["gain"] > p["loss"])(G.paired(a, "native", ("reacher",)))]
    put("SumSigReacherN", len(sig_r))
    put("SumReacherNoLossN", sum(G.paired(a, "native", ("reacher",))["loss"] == 0 for a in locked))
    put("SumReacherSMin", min(v["S"] for v in rc.values()))
    put("SumReacherSMax", max(v["S"] for v in rc.values()))
    ch = [a for a in locked if G.arm_info(a)["adapter"] == "choose"]
    st = [a for a in locked if G.arm_info(a)["adapter"] in ("steer", "stack")]
    put("SumChooseCMin", min(tr[a]["C"] for a in ch))
    put("SumChooseCMax", max(tr[a]["C"] for a in ch))
    put("SumSteerCMin", min(tr[a]["C"] for a in st))
    put("SumSteerCMax", max(tr[a]["C"] for a in st))
    put("SumHarmMin", min(v["harm"] for v in tr.values()))
    put("SumHarmMax", max(v["harm"] for v in tr.values()))
    l4 = {a: G.counts(a, G.ENVS, (4,)) for a in st}
    put("SumLfourVMin", min(v["V"] for v in l4.values()))
    put("SumLfourVMax", max(v["V"] for v in l4.values()))
    l4c = {a: G.counts(a, G.ENVS, (4,)) for a in ch}
    put("SumChooseLfourVMin", min(v["V"] for v in l4c.values()))
    put("SumChooseLfourVMax", max(v["V"] for v in l4c.values()))
    # CMA-ES against random search: pooled training systems and Reacher, per adapter and repeat
    n = s = s_cma = 0
    ch_g = ch_l = 0
    pool = {"Train": [0, 0], "Reacher": [0, 0]}  # CMA-ES cases won / lost against random search, summed over the 6 pairs
    for adapter in ("steer", "choose", "stack"):
        for rep in (1, 2):
            a, b = f"v2-{adapter}-cma-r{rep}", f"v2-{adapter}-rnd-r{rep}"
            if a not in pc or b not in pc:
                continue
            for envs in (G.TRAIN, ("reacher",)):
                p = G.paired(a, b, envs)
                n += 1
                s += p["p"] < 0.05
                s_cma += p["p"] < 0.05 and p["gain"] > p["loss"]
                k = "Reacher" if envs == ("reacher",) else "Train"
                pool[k][0] += p["gain"]
                pool[k][1] += p["loss"]
                if adapter == "choose":
                    ch_g += p["gain"]
                    ch_l += p["loss"]
    put("SumCmaRndTests", n)
    put("SumCmaRndSig", s)
    put("SumCmaRndSigCma", s_cma)
    put("SumChooseCmaGain", ch_g)
    for k, (g, l_) in pool.items():  # not a test: the six pairs share cases and (for Choose + Steer) genes
        put(f"SumCmaRnd{k}Gain", g)
        put(f"SumCmaRnd{k}Loss", l_)
    put("SumChooseCmaLoss", ch_l)
    # untrained draws: gate openings and outcomes identical to native
    for a in [x for x in G.arms(2) if G.arm_info(x)["method"] == "untrained"]:
        i = G.arm_info(a)
        k = "Sum" + i["adapter"].capitalize() + "Untrained" + ["Zero", "One", "Two"][i["draw"]]
        p = pc[a]
        put(k + "Opened", sum(((p[c].get("steer_open") or 0) + (p[c].get("choose_open") or 0)) > 0 for c in p))
        put(k + "Cases", len(p))
        put(k + "SameAsNative", sum(all(p[c].get(f) == nat[c].get(f) for f in ("success", "violation", "min_clear_steps"))
                                    for c in p))
    # native closest approach on levels 2-3, and the range over Steer and Choose picks, TwoRoom
    for env in G.ENVS:
        name = env.capitalize()
        for grp, arms in (("Steer", [a for a in locked if G.arm_info(a)["adapter"] == "steer"]), ("Choose", ch)):
            vals = [G.clearance(a, (env,)) for a in arms]
            put(f"Sum{grp}{name}ClearMin", f"{min(vals):.2f}")
            put(f"Sum{grp}{name}ClearMax", f"{max(vals):.2f}")
    keyed = [G.choose_executed(a) for a in ch]
    put("SumChooseKeyedPctMin", f"{min(100 * k / max(k + o, 1) for k, o in keyed):.0f}")
    put("SumChooseKeyedPctMax", f"{max(100 * k / max(k + o, 1) for k, o in keyed):.0f}")
    avail = [G.choose_clear_available(a) for a in ch]
    put("SumChooseClearAvailMin", f"{min(100 * y / max(m, 1) for y, m in avail):.0f}")
    put("SumChooseClearAvailMax", f"{max(100 * y / max(m, 1) for y, m in avail):.0f}")
    away = [G.steer_push_toward(a) for a in locked if G.arm_info(a)["adapter"] == "steer"]
    put("SumSteerAwayPctMin", f"{min(100 * w / max(t + w, 1) for t, w in away):.0f}")
    c = G.core()
    put("SumGenesSteer", c.n_genes("steer", 2))
    put("SumGenesChoose", c.n_genes("choose", 2))
    lines = ["% Generated by make_summary.py from LeWM_Experiment1/graded/evidence. Do not edit by hand."]
    lines += ["\\newcommand{\\%s}{%s}" % (k, M[k]) for k in sorted(M)]
    (HERE / "summary.tex").write_text("\n".join(lines) + "\n")
    print(f"wrote summary.tex ({len(M)} macros)")
    for k in sorted(M):
        print(k, M[k])


if __name__ == "__main__":
    main()
