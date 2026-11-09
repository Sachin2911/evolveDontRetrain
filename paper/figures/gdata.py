"""Shared evidence loader for paper v2 (graded hazard study, run 1 and, when present, run 2).

Everything the figures, tables and prose numbers use is read from the mirrored evidence:
  LeWM_Experiment1/graded/evidence/{h,c,j}/out/test-*.json      test panel, per case (test-pre.json is superseded)
  LeWM_Experiment1/graded/evidence/{h,c,j}/out/<adapter>-r<k>[-v2]/state.json   searches
  LeWM_Experiment1/graded/evidence/{h,c,j}/prod/<env>/results/*.json            full outcomes (decision records)
  LeWM_Experiment1/graded/panels/<env>.json                                     frozen panels
Run 2 arms are named "v2-..." and their states live in <adapter>-r<k>-v2/. Nothing here assumes they exist.
"""

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[3]
LE = ROOT / "LeWM_Experiment1"
EV = LE / "graded" / "evidence"
PANELS = LE / "graded" / "panels"
BOXES = ("h", "c", "j")
ENVS = ("tworoom", "pusht", "basketball", "reacher")
TRAIN = ("tworoom", "pusht", "basketball")
ENV_NAME = {"tworoom": "TwoRoom", "pusht": "PushT", "basketball": "Basketball", "reacher": "Reacher"}


def level(cid):
    return int(cid.split("-L")[1][0])


def env_of(cid):
    return cid.split("-")[0]


@lru_cache(None)
def panels():
    out = {}
    for env in ENVS:
        for c in json.loads((PANELS / f"{env}.json").read_text()):
            out[c["id"]] = c
    return out


@lru_cache(None)
def per_case():
    """arm -> case_id -> per-case row (success, violation, safe_completion, min_clear_steps, steer_open, choose_open, free)."""
    out = {}
    for f in sorted(EV.glob("*/out/test-*.json")):
        if f.name == "test-pre.json":  # superseded by test-pre2 (first pre-test was restarted)
            continue
        d = json.loads(f.read_text())
        for arm, rows in d["per_case"].items():
            for r in rows:
                out.setdefault(arm, {}).setdefault(r["case_id"], r)
    return out


def arms(version=None):
    names = sorted(per_case())
    if version is None:
        return names
    return [a for a in names if (a.startswith("v2-")) == (version == 2)]


def arm_info(name):
    """-> dict(version, adapter, method, repeat, draw, veto)."""
    v = 2 if name.startswith("v2-") else 1
    base = name[3:] if v == 2 else name
    if base == "native":
        return dict(version=v, adapter="native", method="native", repeat=None, draw=None, veto=False)
    parts = base.split("-")
    adapter = parts[0]
    if parts[1].startswith("untrained"):
        return dict(version=v, adapter=adapter, method="untrained", repeat=None, draw=int(parts[1][9:]), veto=False)
    if parts[1] == "nearzero":
        return dict(version=v, adapter=adapter, method="nearzero", repeat=None, draw=None, veto=False)
    return dict(version=v, adapter=adapter, method=parts[1], repeat=int(parts[2][1:]), draw=None,
                veto=len(parts) > 3 and parts[3] == "veto")


def cases_for(envs, levels):
    return sorted(c for c, v in panels().items() if v["split"] == "test" and v["env"] in envs and v["level"] in levels)


def counts(arm, envs=ENVS, levels=(1, 2, 3)):
    """S, C, V, harm (native safe, arm not) and n over the test cases of envs x levels that the arm ran."""
    pc, nat = per_case()[arm], per_case()["native"]
    cs = [c for c in cases_for(envs, levels) if c in pc]
    S = sum(bool(pc[c]["safe_completion"]) for c in cs)
    C = sum(bool(pc[c]["success"]) for c in cs)
    V = sum(bool(pc[c]["violation"]) for c in cs)
    harm = sum(bool(nat[c]["safe_completion"]) and not pc[c]["safe_completion"] for c in cs)
    return dict(S=S, C=C, V=V, harm=harm, n=len(cs))


def paired(a, b, envs=ENVS, levels=(1, 2, 3)):
    """Case-paired safe completion: gains (a safe, b not), losses (b safe, a not), two-sided exact sign-test p."""
    pa, pb = per_case()[a], per_case()[b]
    cs = [c for c in cases_for(envs, levels) if c in pa and c in pb]
    g = sum(bool(pa[c]["safe_completion"]) and not pb[c]["safe_completion"] for c in cs)
    lo = sum(bool(pb[c]["safe_completion"]) and not pa[c]["safe_completion"] for c in cs)
    p = binomtest(g, g + lo).pvalue if g + lo else 1.0
    return dict(gain=g, loss=lo, p=float(p), n=len(cs))


def clearance(arm, envs, levels=(2, 3)):
    pc = per_case()[arm]
    cs = [c for c in cases_for(envs, levels) if c in pc]
    return float(np.mean([float(pc[c]["min_clear_steps"] or 0.0) for c in cs])) if cs else float("nan")


def gate_open(arm, envs, levels=(2, 3)):
    pc = per_case()[arm]
    cs = [c for c in cases_for(envs, levels) if c in pc]
    k = sum((pc[c].get("steer_open") or 0) + (pc[c].get("choose_open") or 0) > 0 for c in cs)
    return k, len(cs)


@lru_cache(None)
def states(version=1):
    """(adapter, repeat) -> state dict. The locked states were copied to every box; keep the longest copy."""
    suffix = "-v2" if version == 2 else ""
    out = {}
    for f in sorted(EV.glob("*/out/*-r*/state.json")):
        s = json.loads(f.read_text())
        if not s.get("generations") or f.parent.name != f"{s['adapter']}-r{s['repeat']}{suffix}":
            continue
        key = (s["adapter"], s["repeat"])
        if key not in out or (len(s["generations"]), "locked" in s) > (len(out[key]["generations"]), "locked" in out[key]):
            out[key] = s
    return out


def decode(x, adapter):
    """Decode a locked genotype with the study's own code (LeWM_Experiment1/graded/core.py, read-only)."""
    import sys
    if str(LE) not in sys.path:
        sys.path.insert(0, str(LE))
    from graded import core
    return core.decode(np.asarray(x, float), adapter)


def core():
    import sys
    if str(LE) not in sys.path:
        sys.path.insert(0, str(LE))
    from graded import core as c
    return c


def results(arm, env):
    """Full outcomes (with decision records) of an arm's test jobs on one env, one per case."""
    out = {}
    pat = f"*-test-{arm}-{env}-*.json"
    for f in sorted(EV.glob(f"*/prod/{env}/results/{pat}")):
        if f.name.endswith("-wd.json"):
            continue
        cid = f.name.split(f"-test-{arm}-", 1)[1][: -len(".json")]
        if cid in out:
            continue
        try:
            out[cid] = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
    return out


def steer_push_toward(arm, envs=("tworoom", "pusht", "reacher"), levels=(2, 3)):
    """Over open Steer decisions on AP envs: how many pushed sideways toward the hazard's side.

    side sense > 0 means the hazard is on the left of travel; sideways > 0 moves left. Toward = same sign.
    """
    toward = away = 0
    for env in envs:
        for cid, o in results(arm, env).items():
            if level(cid) not in levels:
                continue
            for r in o.get("steer_records") or []:
                if not r.get("open") or "sideways" not in r:
                    continue
                side, sw = r["senses"][1], r["sideways"]
                if abs(sw) < 1e-9 or side == 0:
                    continue
                if np.sign(side) == np.sign(sw):
                    toward += 1
                else:
                    away += 1
    return toward, away


def steer_pushed(arm, envs=("tworoom", "pusht", "reacher"), levels=(2, 3)):
    """Over open Steer decisions on AP envs: how many actually pushed (non-zero sideways), and the mean push size."""
    pushed = opened = 0
    sizes = []
    for env in envs:
        for cid, o in results(arm, env).items():
            if level(cid) not in levels:
                continue
            for r in o.get("steer_records") or []:
                if not r.get("open"):
                    continue
                opened += 1
                sw = abs(float(r.get("sideways", 0.0)))
                if sw > 1e-9:
                    pushed += 1
                    sizes.append(sw)
    return pushed, opened, (float(np.mean(sizes)) if sizes else 0.0)


def choose_executed(arm, envs=("tworoom", "pusht", "reacher"), levels=(2, 3)):
    """Over open Choose decisions on AP envs: how many executed the safety-keyed plan vs the planner's own."""
    keyed = own = 0
    for env in envs:
        for cid, o in results(arm, env).items():
            if level(cid) not in levels:
                continue
            for r in o.get("choose_records") or []:
                if not r.get("open"):
                    continue
                if str(r.get("executed", "")).startswith("keyed"):
                    keyed += 1
                else:
                    own += 1
    return keyed, own


def choose_clear_available(arm, envs=("tworoom", "pusht", "reacher"), levels=(2, 3)):
    """Over open Choose decisions: how many had at least one checked candidate predicted clear of the hazard."""
    yes = n = 0
    for env in envs:
        for cid, o in results(arm, env).items():
            if level(cid) not in levels:
                continue
            for r in o.get("choose_records") or []:
                if not r.get("open") or not r.get("rounds"):
                    continue
                n += 1
                yes += max(x.get("best_clear", -1e9) for x in r["rounds"]) > 0
    return yes, n


def fmt_p(p):
    if p >= 0.995:
        return "1.00"
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}" if p < 0.1 else f"{p:.2f}"
