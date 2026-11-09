"""Every number in paper v2's prose and generated tables, computed from the mirrored evidence.

  python paper/make_numbers.py

Writes numbers.tex (one \\newcommand per number) and tables/*.tex. Run-2 macros and tables (v2 arms, states in
<adapter>-r<k>-v2/) are written only when that evidence exists; numbers.tex then also defines \\runtwotrue.
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "figures"))
import gdata as G  # noqa: E402

DIG = dict(zip("0123456789", ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]))
M = {}


def name(*parts):
    out = ""
    for p in parts:
        for tok in str(p).replace("_", "-").split("-"):
            tok = "".join(DIG.get(ch, ch) for ch in tok)
            out += tok[:1].upper() + tok[1:]
    assert out.isalpha(), out
    return out


def put(key, value):
    M[key] = value


def f2(x):
    return f"{x:.2f}"


def f1(x):
    return f"{x:.1f}"


def pstr(p):
    return G.fmt_p(p)


# ------------------------------------------------------------------ panels and units

def panel_numbers():
    pan = G.panels()
    summ = json.loads((G.PANELS / "summary.json").read_text())
    for env in G.ENVS:
        cs = [c for c in pan.values() if c["env"] == env]
        put(name("step", env), {"tworoom": f"{cs[0]['step']:.1f}", "pusht": f"{cs[0]['step']:.1f}",
                                "reacher": f"{100 * cs[0]['step']:.2f}", "basketball": f"{100 * cs[0]['step']:.2f}"}[env])
        put(name("body", env), {"tworoom": "7", "pusht": "15", "reacher": "1", "basketball": "0"}[env])
        for lv in (2, 3):
            d = [c["native"]["min_clear_steps"] for c in cs if c["level"] == lv]
            put(name("native", "depth", env, "L", lv), f2(-float(np.mean(d))))
        d1 = [c["native"]["min_clear_steps"] for c in cs if c["level"] == 1]
        put(name("native", "clear", env, "Lone", "min"), f1(float(np.min(d1))))
        put(name("native", "clear", env, "Lone", "max"), f1(float(np.max(d1))))
    put("nTrainPerEnv", sum(1 for c in pan.values() if c["env"] == "tworoom" and c["split"] == "train"))
    put("nSelectPerEnv", sum(1 for c in pan.values() if c["env"] == "tworoom" and c["split"] == "select"))
    put("nTestPerEnv", sum(1 for c in pan.values() if c["env"] == "tworoom" and c["split"] == "test"))
    put("nTestAll", sum(1 for c in pan.values() if c["split"] == "test"))
    put("nSelectAll", sum(1 for c in pan.values() if c["split"] == "select"))
    put("nTrainAll", sum(1 for c in pan.values() if c["split"] == "train"))


# ------------------------------------------------------------------ searches

def search_numbers(version):
    import importlib
    if str(G.LE) not in sys.path:
        sys.path.insert(0, str(G.LE))
    stoprule = importlib.import_module("graded.stoprule")
    pre = "vtwo" if version == 2 else ""
    total = 0
    for (adapter, rep), s in sorted(G.states(version).items()):
        gens = s["generations"]
        k = name(pre, adapter, "r", rep)
        n_cases = len(gens[0]["cases"])
        jobs = sum(g["jobs"] for g in gens)
        total += jobs
        put(k + "Gens", len(gens))
        put(k + "Dim", len(gens[0]["cma_x"][0]))
        put(k + "Pop", len(gens[0]["cma_x"]))
        put(k + "CasesPerGen", n_cases)
        put(k + "Jobs", jobs)
        free = np.mean([c.get("free", 0) / n_cases for g in gens for arm in ("cma", "rnd") for c in g[arm]])
        put(k + "FreePct", f"{100 * free:.0f}")
        for arm in ("cma", "rnd"):
            d = np.array([[c["S"] - g["native"]["S"] for c in g[arm]] for g in gens], float)
            put(k + name(arm, "mean", "dS"), f2(float(d.mean())))
            put(k + name(arm, "best", "dS"), f2(float(d.max(1).mean())))
            harm = np.mean([c["harm"] for g in gens for c in g[arm]])
            put(k + name(arm, "harm"), f2(float(harm)))
        full = stoprule.check(Path(G.EV) / "x", None, 1) if False else None  # noqa: F841 (placeholder, see below)
        out = check_state(stoprule, s, None)
        put(k + "TrendCma", f2(out["trend_cma"][0]))
        put(k + "TrendCmaP", pstr(out["trend_cma"][1]))
        put(k + "TrendRnd", f2(out["trend_rnd"][0]))
        put(k + "TrendRndP", pstr(out["trend_rnd"][1]))
        if len(gens) >= 8:
            e = check_state(stoprule, s, 8)
            put(k + "EightTrendCma", f2(e["trend_cma"][0]))
            put(k + "EightTrendCmaP", pstr(e["trend_cma"][1]))
            put(k + "EightTrendRnd", f2(e["trend_rnd"][0]))
            put(k + "EightMoved", f2(e["moved"]))
            put(k + "EightNullQ", f2(e["null_q95"]))
            put(k + "EightPass", "passed" if e["pass"] else "failed")
        lk = s.get("locked", {})
        for arm in ("cma", "rnd"):
            if arm in lk:
                put(k + name("sel", arm, "S"), lk[arm]["S"])
                put(k + name("sel", arm, "C"), lk[arm]["C"])
                put(k + name("sel", arm, "harm"), lk[arm]["harm"])
                put(k + name("sel", arm, "gen"), lk[arm]["id"].split("-")[2][1:] if lk[arm]["id"].count("-") >= 3 else "mean")
        if "native_selection" in s:
            put(k + "SelNativeS", int(s["native_selection"][1]))
            put(k + "SelNativeC", int(s["native_selection"][2]))
    put(name(pre, "search", "jobs", "total"), f"{total:,}".replace(",", "{,}"))


def check_state(stoprule, s, n_gens, nulls=200):
    """stoprule.check on an in-memory state (the module reads from a path)."""
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(s, fh)
        path = fh.name
    try:
        return stoprule.check(path, n_gens, nulls)
    finally:
        Path(path).unlink()


# ------------------------------------------------------------------ test panel

def test_numbers(version):
    pre = "vtwo" if version == 2 else ""
    arms = [a for a in G.arms(version) if a != "native"] + (["native"] if version == 1 else [])
    for a in arms:
        k = name(pre, a[3:] if a.startswith("v2-") else a)
        for grp, envs in (("Train", G.TRAIN), ("Reacher", ("reacher",)), ("All", G.ENVS)):
            c = G.counts(a, envs)
            put(k + grp + "S", c["S"])
            put(k + grp + "C", c["C"])
            put(k + grp + "V", c["V"])
            put(k + grp + "Harm", c["harm"])
            if a != "native":
                p = G.paired(a, "native", envs)
                put(k + grp + "Gain", p["gain"])
                put(k + grp + "Loss", p["loss"])
                put(k + grp + "P", pstr(p["p"]))
        for env in G.ENVS:
            c = G.counts(a, (env,))
            put(k + name(env) + "S", c["S"])
        c4 = G.counts(a, G.ENVS, (4,))
        put(k + "LfourC", c4["C"])
        put(k + "LfourV", c4["V"])
        put(k + "LfourClean", c4["n"] - c4["V"])
    # CMA vs random, per adapter and repeat
    for adapter in ("steer", "choose", "stack"):
        for rep in (1, 2):
            a = ("v2-" if version == 2 else "") + f"{adapter}-cma-r{rep}"
            b = ("v2-" if version == 2 else "") + f"{adapter}-rnd-r{rep}"
            if a in G.per_case() and b in G.per_case():
                for grp, envs in (("Train", G.TRAIN), ("Reacher", ("reacher",))):
                    p = G.paired(a, b, envs)
                    k = name(pre, adapter, "cma", "vs", "rnd", "r", rep, grp)
                    put(k + "Gain", p["gain"])
                    put(k + "Loss", p["loss"])
                    put(k + "P", pstr(p["p"]))
    # how many paired tests were run, and how many came out below 0.05 (all arms vs native, CMA vs random, per env)
    locked = [a for a in G.arms(version) if G.arm_info(a)["method"] in ("cma", "rnd") and not G.arm_info(a)["veto"]]
    tests = []
    for a in locked:
        for env in G.ENVS:
            tests.append((a, "native", env, G.paired(a, "native", (env,))))
    for a in locked:
        i = G.arm_info(a)
        if i["method"] == "cma":
            b = a.replace("-cma-", "-rnd-")
            if b in G.per_case():
                for env in G.ENVS:
                    tests.append((a, b, env, G.paired(a, b, (env,))))
    sig = [t for t in tests if t[3]["p"] < 0.05]
    tj = 0
    for f in sorted(G.EV.glob("*/out/test-*.json")):
        if f.name == "test-pre.json":
            continue
        d = json.loads(f.read_text())
        if any(x.startswith("v2-") for x in d["arms"]) == (version == 2):
            tj += d["jobs"]
    put(name(pre, "test", "jobs", "total"), f"{tj:,}".replace(",", "{,}"))
    put(name(pre, "n", "paired", "tests"), len(tests))
    put(name(pre, "n", "paired", "sig"), len(sig))
    cma_beats_rnd = [t for t in tests if t[1] != "native" and t[3]["p"] < 0.05 and t[3]["gain"] > t[3]["loss"]]
    put(name(pre, "n", "cma", "beats", "rnd", "sig"), len(cma_beats_rnd))
    # level 4: violations avoided and completions given up
    nat4 = G.counts("native", G.ENVS, (4,))
    put("nativeLfourC", nat4["C"])
    put("nativeLfourV", nat4["V"])
    l4 = {a: G.counts(a, G.ENVS, (4,)) for a in G.arms(version) if a != "native"}
    if l4:
        best = min(l4, key=lambda a: (l4[a]["V"], -l4[a]["C"]))
        put(name(pre, "Lfour", "fewest", "V"), l4[best]["V"])
        put(name(pre, "Lfour", "fewest", "V", "C"), l4[best]["C"])
        put(name(pre, "Lfour", "fewest", "V", "arm"), short(best))
        put(name(pre, "Lfour", "max", "S"), max(G.counts(a, G.ENVS, (4,))["S"] for a in G.arms(version)))
    # gate and clearance diagnosis
    for a in locked:
        k = name(pre, a[3:] if a.startswith("v2-") else a)
        o, n = gate_open_all(a)
        put(k + "OpenLtt", o)
        put(k + "OpenLttN", n)
        for env in G.ENVS:
            put(k + name(env) + "Clear", f2(G.clearance(a, (env,))))
        i = G.arm_info(a)
        if i["adapter"] == "choose":
            kk, own = G.choose_executed(a)
            put(k + "Keyed", kk)
            put(k + "OwnPlan", own)
            put(k + "KeyedPct", f"{100 * kk / max(kk + own, 1):.0f}")
            y, n = G.choose_clear_available(a)
            put(k + "ClearAvail", y)
            put(k + "ClearAvailN", n)
            put(k + "ClearAvailPct", f"{100 * y / max(n, 1):.0f}")
        if i["adapter"] == "steer":
            t, w = G.steer_push_toward(a)
            put(k + "Away", w)
            put(k + "Toward", t)
            put(k + "AwayPct", f"{100 * w / max(t + w, 1):.0f}")
            size, cap = push_size(a)
            put(k + "PushSize", f2(size))
            put(k + "PushCap", f2(cap))
    for env in G.ENVS:
        put(name("native", env, "Clear"), f2(G.clearance("native", (env,))))


def short(a):
    i = G.arm_info(a)
    if i["method"] == "untrained":
        return f"untrained {i['adapter'].capitalize()} draw {i['draw'] + 1}"
    meth = {"cma": "CMA-ES", "rnd": "random-search", "nearzero": "near-zero"}[i["method"]]
    rep = f" (repeat {i['repeat']})" if i["repeat"] else ""
    return f"{meth} {i['adapter'].capitalize()}{' v2' if i['version'] == 2 else ''}{rep}"


def gate_open_all(a):
    o = n = 0
    for env in G.ENVS:
        k, m = G.gate_open(a, (env,))
        o += k
        n += m
    return o, n


def push_size(a):
    """Mean |sideways| (steps) over open AP decisions on levels 2-3, and the nominee's authority cap."""
    sw = []
    for env in ("tworoom", "pusht", "reacher"):
        for cid, o in G.results(a, env).items():
            if G.level(cid) in (2, 3):
                sw += [abs(r["sideways"]) for r in (o.get("steer_records") or []) if r.get("open") and "sideways" in r]
    i = G.arm_info(a)
    st = G.states(i["version"])[("steer", i["repeat"])]
    g = G.decode(st["locked"][i["method"]]["x"], "steer")
    return (float(np.mean(sw)) if sw else float("nan")), g["authority"]


def gene_numbers():
    pre_ = {1: "", 2: "vtwo"}
    for version in (1, 2):
        st = G.states(version)
        for (adapter, rep), s in st.items():
            if "locked" not in s:
                continue
            for arm in ("cma", "rnd"):
                g = G.decode(s["locked"][arm]["x"], adapter)
                k = name(pre_[version], adapter, arm, "r", rep)
                for gene in ("exec", "authority", "gamma", "margin", "lam", "gain", "factor", "behind", "release"):
                    if gene in g:
                        put(k + name("gene", gene), f2(g[gene]))


# ------------------------------------------------------------------ generated tables

def table_levels(version):
    rows = [("Planner alone", "native")]
    pre = "v2-" if version == 2 else ""
    for adapter, lab in (("steer", "Steer"), ("choose", "Choose"), ("stack", "Choose + Steer")):
        for m, ml in (("cma", "CMA-ES"), ("rnd", "random search")):
            for rep in (1, 2):
                a = f"{pre}{adapter}-{m}-r{rep}"
                if a in G.per_case():
                    rows.append((f"{lab}, {ml}, repeat {rep}", a))
        unt = sorted(a for a in G.per_case() if a.startswith(f"{pre}{adapter}-untrained"))
        for a in unt:
            rows.append((f"{lab}, untrained adapter {G.arm_info(a)['draw'] + 1}", a))
    lines = []
    nat = {(env, lv): G.counts("native", (env,), (lv,))["S"] for env in G.ENVS for lv in (1, 2, 3)}
    prev_family = None
    for lab, a in rows:
        family = lab.split(",")[0]
        if prev_family not in (None, "Planner alone") and family != prev_family:
            lines.append(r"\addlinespace[2pt]")
        prev_family = family
        cells = []
        for env in G.ENVS:
            for lv in (1, 2, 3):
                v = G.counts(a, (env,), (lv,))["S"]
                n = nat[(env, lv)]
                # above native = bold, below native = grey; level 4 shows completions/violations
                cells.append(f"\\textbf{{{v}}}" if (a != "native" and v > n) else
                             (f"\\textcolor{{gray}}{{{v}}}" if v < n else str(v)))
            c4 = G.counts(a, (env,), (4,))
            cells.append(f"{c4['C']}/{c4['V']}")
        bold = "\\textbf{%s}" % lab if a == "native" else lab
        lines.append(bold + " & " + " & ".join(cells) + r" \\")
        if a == "native":
            lines.append(r"\midrule")
    Path(HERE / "tables").mkdir(exist_ok=True)
    out = HERE / "tables" / ("test_levels_run1.tex" if version == 1 else "test_levels.tex")
    out.write_text("\n".join(lines) + "\n")
    print("wrote", out.name)


def table_paired(version):
    pre = "v2-" if version == 2 else ""
    lines = []
    for adapter, lab in (("steer", "Steer"), ("choose", "Choose"), ("stack", "Choose + Steer")):
        for rep in (1, 2):
            a, b = f"{pre}{adapter}-cma-r{rep}", f"{pre}{adapter}-rnd-r{rep}"
            if a not in G.per_case() or b not in G.per_case():
                continue
            cells = []
            for x, y in ((a, "native"), (b, "native"), (a, b)):
                for envs in (G.TRAIN, ("reacher",)):
                    p = G.paired(x, y, envs)
                    s = f"{p['gain']}--{p['loss']}"
                    cells.append(s)
                    cells.append(("\\textbf{%s}" % pstr(p["p"])) if p["p"] < 0.05 else pstr(p["p"]))
            lines.append(f"{lab}, repeat {rep} & " + " & ".join(cells) + r" \\")
        lines.append(r"\addlinespace")
    if lines and lines[-1] == r"\addlinespace":
        lines.pop()
    out = HERE / "tables" / ("paired_run1.tex" if version == 1 else "paired.tex")
    out.write_text("\n".join(lines) + "\n")
    print("wrote", out.name)


def table_genes():
    c = G.core()
    plain = {"release": r"release $\rho$", "margin": r"margin $m$", "gain": r"gain $\kappa$", "authority": r"authority $A$",
             "factor": r"slow-down $f$", "behind": r"behind $h$", "lam": r"penalty $\lambda$",
             "r0": r"first round $r_0$", "r1": r"last round $r_1$"}  # the paper's symbols (Eqs. 4-6, Algorithm 1)

    def tex(nm):
        return plain.get(nm, nm.replace("_", r"\_"))
    # the paper's Steer and Choose are the version-2 designs; the version-1 boxes are kept under *_run1 for the record
    files = {"steer": "genes_steer_run1", "choose": "genes_choose_run1", "steer2": "genes_steer", "choose2": "genes_choose"}
    for key, label in (("steer", "Steer"), ("choose", "Choose"), ("steer2", "Steer"), ("choose2", "Choose")):
        names, lo, hi = c.BOXES[key]
        groups, seen = [], set()
        for n_, a, b in zip(names, lo, hi):
            if n_.startswith("trig_"):
                if "trig" in seen:
                    continue
                seen.add("trig")
                groups.append((r"gate $w_k$, $b$ (9)", -3.0, 3.0))
            elif n_.startswith(("u_", "v_", "f_")) and n_.split("_")[0] in seen:
                continue
            elif n_.startswith(("u_", "v_", "f_")):
                pfx = n_.split("_")[0]
                seen.add(pfx)
                cnt = sum(1 for m in names if m.startswith(pfx + "_"))
                groups.append((f"{pfx}" + r"\_$k$" + f", {pfx}" + r"\_bias" + f" ({cnt})", a, b))
            else:
                groups.append((tex(n_), a, b))
        M[name("genes", key)] = len(names)
        rows = [f"{g} & [{a:g}, {b:g}]" + r" \\" for g, a, b in groups]
        (HERE / "tables" / f"{files[key]}.tex").write_text("\n".join(rows) + "\n")
    print("wrote genes tables")


def main():
    (HERE / "tables").mkdir(exist_ok=True)
    panel_numbers()
    search_numbers(1)
    test_numbers(1)
    gene_numbers()
    table_levels(1)
    table_paired(1)
    table_genes()
    run2 = bool(G.states(2)) or bool(G.arms(2))
    if G.states(2):
        search_numbers(2)
    if G.arms(2):
        test_numbers(2)
        table_levels(2)
        table_paired(2)
    lines = ["% Generated by make_numbers.py from LeWM_Experiment1/graded/evidence. Do not edit by hand.",
             r"\newif\ifruntwo", r"\runtwotrue" if run2 else r"\runtwofalse"]
    for k in sorted(M):
        lines.append("\\newcommand{\\%s}{%s}" % (k, M[k]))
    (HERE / "numbers.tex").write_text("\n".join(lines) + "\n")
    print(f"wrote numbers.tex ({len(M)} macros, run 2 {'present' if run2 else 'absent'})")


if __name__ == "__main__":
    main()
