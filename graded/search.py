"""Graded study driver (9 Nov 2026, 4-hour run). Runs on a host's CPU; talks to workers through env queues.

Modes:
  natives   run native (log_senses) on every case of the given splits; writes <out>/natives-<env>.json
  search    one repeat of one adapter: CMA-ES (diagonal, bounded [-1, 1], x0 = box centre = native) and uniform
            random search in LOCKSTEP (same generation count, same 9 rotating cases per generation: one per
            training env per level 1-3), until --max-gens or the --deadline (UTC epoch), at least --min-gens.
            Then nominees (4 per method: random top 4; CMA top 3 + its mean, by improvement over native on their own
            cases), re-scored on the 18-case selection panel; one locked per method.
  test      every arm on the test panel of the given envs (24 cases each), with free outcomes.

Score (larger is better, compared in order): (-harm, S, C, -contact depth, -goal distance, -trigger decisions).
harm = cases where native completed safely and the candidate did not (Deb: feasibility first).
Plan: docs/research/reports/Graded hazard study plan.md; cuts for the 4-hour run: graded/PREREG.md.
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np

from graded import core

TRAIN_ENVS = ("tworoom", "pusht", "basketball")


def read(p):
    return json.loads(Path(p).read_text())


def write(p, obj):
    p = Path(p)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, default=float))
    tmp.rename(p)


class Queues:
    def __init__(self, root):
        self.root = Path(root)

    def submit(self, job):
        d = self.root / job["env"] / "jobs"
        d.mkdir(parents=True, exist_ok=True)
        (self.root / job["env"] / "results").mkdir(parents=True, exist_ok=True)
        write(d / (job["job_id"] + ".json"), job)

    def wait(self, jobs, poll=2.0, log=None):
        out = {}
        while len(out) < len(jobs):
            for job in jobs:
                jid = job["job_id"]
                if jid in out:
                    continue
                p = self.root / job["env"] / "results" / (jid + ".json")
                if p.exists():
                    try:
                        out[jid] = read(p)
                    except json.JSONDecodeError:
                        continue
                    if log is not None:
                        slim = {k: v for k, v in out[jid].items() if k not in ("senses_prop", "senses_warm", "body", "trace", "records")}
                        log.write(json.dumps(slim, default=float) + "\n")
                        log.flush()
            if len(out) < len(jobs):
                time.sleep(poll)
        return out


def load_panels(panel_dir, envs):
    cases = {}
    for env in envs:
        for c in read(Path(panel_dir) / f"{env}.json"):
            cases[c["id"]] = c
    return cases


def load_natives(out_dir, envs):
    nat = {}
    for env in envs:
        p = Path(out_dir) / f"natives-{env}.json"
        if p.exists():
            nat.update(read(p))
    return nat


# ------------------------------------------------------------------ scoring

def case_terms(o, nat):
    ok = bool(o.get("completed"))
    safe = ok and bool(o.get("safe_completion"))
    done = ok and bool(o.get("success"))
    harm = int(bool(nat.get("safe_completion")) and not safe)
    depth = max(0.0, -float(o.get("min_clear_steps", 0.0))) if ok else 6.0
    goal = float(o.get("final_goal_steps", 6.0)) if ok else 99.0
    trig = int(o.get("steer_open", 0)) + int(o.get("choose_open", 0))
    return harm, int(safe), int(done), depth, goal, trig, int(not ok)


def score(outs, natives):
    """outs: list of (outcome, native outcome). Returns the score tuple (larger is better) and its parts."""
    t = np.array([case_terms(o, n) for o, n in outs], float)
    harm, S, C, depth, goal, trig, errors = t.sum(axis=0)
    tup = (-harm, S, C, -round(depth, 4), -round(goal, 4), -trig)
    return tup, {"harm": int(harm), "S": int(S), "C": int(C), "depth": float(depth), "goal": float(goal),
                 "trig": int(trig), "errors": int(errors)}


def native_score(case_ids, natives):
    return score([(natives[c], natives[c]) for c in case_ids], natives)


def improvement(tup, nat_tup):
    return tuple(a - b for a, b in zip(tup, nat_tup))


def ranks_of(tuples):
    order = sorted(range(len(tuples)), key=lambda i: (tuple(-v for v in tuples[i]), i))
    r = [0] * len(tuples)
    for k, i in enumerate(order):
        r[i] = k
    return r


# ------------------------------------------------------------------ jobs with free outcomes

def free_outcome(adapter, x, nat):
    """Exact free outcome: the trigger never opens on the native run's logged senses -> the run IS native."""
    if adapter == "steer":
        if nat.get("senses_prop") is None:
            return False
        return core.steer_never_opens(x, nat["senses_prop"])
    if adapter == "choose":
        if nat.get("senses_warm") is None:
            return False
        return core.choose_never_opens(x, nat["senses_warm"])
    return False


def make_job(job_id, case, adapter, xs, veto=False):
    job = {"job_id": job_id, "env": case["env"], "kind": adapter, "case": case, "log_senses": False}
    if adapter in ("steer", "stack"):
        job["x_steer"] = [float(v) for v in xs["steer"]]
        job["veto"] = bool(veto)
    if adapter in ("choose", "stack"):
        job["x_choose"] = [float(v) for v in xs["choose"]]
    return job


def evaluate(q, members, case_ids, cases, natives, tag, log, free=True):
    """members: list of (member_id, adapter, xs dict, veto). Returns {member_id: [(outcome, native), ...]}."""
    jobs, plan = [], {}
    for mid, adapter, xs, veto in members:
        plan[mid] = []
        for cid in case_ids:
            nat = natives[cid]
            if adapter == "native":
                plan[mid].append((cid, None))
                continue
            single = adapter in ("steer", "choose")
            if free and single and not veto and free_outcome(adapter, xs[adapter], nat):
                plan[mid].append((cid, None))
                continue
            if free and adapter == "stack" and free_outcome("steer", xs["steer"], nat) and free_outcome("choose", xs["choose"], nat):
                plan[mid].append((cid, None))
                continue
            job = make_job(f"{tag}-{mid}-{cid}", cases[cid], adapter, xs, veto)
            q.submit(job)
            jobs.append(job)
            plan[mid].append((cid, job["job_id"]))
    res = q.wait(jobs, log=log) if jobs else {}
    out = {}
    for mid, items in plan.items():
        out[mid] = []
        for cid, jid in items:
            o = dict(natives[cid], free=True) if jid is None else res[jid]
            out[mid].append((o, natives[cid]))
    return out, len(jobs)


# ------------------------------------------------------------------ modes

def mode_natives(args, q):
    envs = args.envs.split(",")
    cases = load_panels(args.panels, envs)
    splits = set(args.splits.split(","))
    jobs = [{"job_id": f"{args.tag}-nat-{cid}", "env": c["env"], "kind": "native", "case": c, "log_senses": True}
            for cid, c in cases.items() if c["split"] in splits]
    for j in jobs:
        q.submit(j)
    print(f"natives: {len(jobs)} jobs", flush=True)
    log = (Path(args.out) / f"episodes-natives-{args.tag}.jsonl").open("a")
    res = q.wait(jobs, log=log)
    for env in envs:
        d = {j["case"]["id"]: res[j["job_id"]] for j in jobs if j["env"] == env}
        p = Path(args.out) / f"natives-{env}.json"
        old = read(p) if p.exists() else {}
        old.update(d)
        write(p, old)
        bad = [k for k, v in d.items() if not v.get("completed")]
        safe = sum(bool(v.get("safe_completion")) for v in d.values())
        print(f"{env}: {len(d)} natives, {safe} safe, {len(bad)} errors {bad[:3]}", flush=True)


def by_split_level(cases, env, split):
    out = {}
    for c in cases.values():
        if c["env"] == env and c["split"] == split:
            out.setdefault(c["level"], []).append(c["id"])
    for k in out:
        out[k].sort()
    return out


def mode_search(args, q):
    import cma

    adapter = args.adapter
    n = core.n_genes(adapter, args.version)
    train_envs = tuple(args.train_envs.split(","))
    cases = load_panels(args.panels, train_envs)
    natives = load_natives(args.out, train_envs)
    pools = {env: by_split_level(cases, env, "train") for env in train_envs}
    select_ids = sorted(c for c, v in cases.items() if v["split"] == "select")
    out = Path(args.out) / f"{adapter}-r{args.repeat}{args.variant}"
    out.mkdir(parents=True, exist_ok=True)
    state_p = out / "state.json"
    state = read(state_p) if state_p.exists() else {"generations": [], "adapter": adapter, "repeat": args.repeat}
    log = (out / "episodes.jsonl").open("a")
    seed = args.seed + 1000 * args.repeat + (0 if adapter == "steer" else 500) + (50 if args.version == 2 else 0)
    es = cma.CMAEvolutionStrategy(np.zeros(n), args.sigma0, {
        "popsize": args.pop, "CMA_diagonal": True, "bounds": [-1.0, 1.0], "seed": seed, "verbose": -9,
        "verb_log": 0, "verb_disp": 0, "tolfun": 0, "tolfunhist": 0, "tolflatfitness": 10**6, "tolx": 0,
        "maxiter": 10**6})
    for g in state["generations"]:  # resume
        es.ask()
        es.tell([np.asarray(x) for x in g["cma_x"]], g["cma_ranks"])
    rrng = np.random.default_rng(seed + 7 + len(state["generations"]))
    gen_seconds = []
    while len(state["generations"]) < args.max_gens:
        gen = len(state["generations"])
        est = float(np.median(gen_seconds)) if gen_seconds else 0.0
        if gen >= args.min_gens and time.time() + est > args.deadline:
            print(f"deadline: stop after {gen} generations", flush=True)
            break
        t0 = time.time()
        case_ids = []
        fixed = [int(v) for v in args.fixed.split(",")] if args.fixed else None
        for env in train_envs:
            for level in (1, 2, 3):
                ids = pools[env][level]
                if fixed:  # run 2: the same cases every generation; repeats take disjoint subsets where possible
                    k = fixed[level - 1]
                    case_ids.extend(ids[((args.repeat - 1) * k + j) % len(ids)] for j in range(k))
                elif args.all_cases:
                    case_ids.extend(ids)
                else:
                    case_ids.append(ids[(gen + level) % len(ids)])
        cma_x = [np.clip(np.asarray(x), -1, 1) for x in es.ask()]
        rnd_x = [rrng.uniform(-1.0, 1.0, n) for _ in range(args.pop)]
        members = [(f"cma{i:02d}", adapter, {adapter: x}, False) for i, x in enumerate(cma_x)]
        members += [(f"rnd{i:02d}", adapter, {adapter: x}, False) for i, x in enumerate(rnd_x)]
        res, n_jobs = evaluate(q, members, case_ids, cases, natives, f"{args.tag}-{adapter}-r{args.repeat}{args.variant}-g{gen:02d}", log)
        nat_tup, nat_parts = native_score(case_ids, natives)
        rec = {"generation": gen, "cases": case_ids, "native": {"score": nat_tup, **nat_parts}, "jobs": n_jobs,
               "cma_x": [x.tolist() for x in cma_x], "rnd_x": [x.tolist() for x in rnd_x], "cma": [], "rnd": []}
        for mid, _, xs, _ in members:
            tup, parts = score(res[mid], natives)
            arm = "cma" if mid.startswith("cma") else "rnd"
            rec[arm].append({"id": mid, "score": tup, "improvement": improvement(tup, nat_tup), **parts,
                             "free": int(sum(1 for o, _ in res[mid] if o.get("free")))})
        rec["cma_ranks"] = ranks_of([tuple(c["score"]) for c in rec["cma"]])
        es.tell([np.asarray(x) for x in cma_x], rec["cma_ranks"])
        rec["cma_mean"] = np.clip(es.mean, -1, 1).tolist()
        rec["sigma"] = float(es.sigma)
        rec["seconds"] = time.time() - t0
        gen_seconds.append(rec["seconds"])
        state["generations"].append(rec)
        write(state_p, state)
        best = lambda arm: max(tuple(c["score"]) for c in rec[arm])
        print(f"[{adapter} r{args.repeat}] gen {gen}: cma best {best('cma')} rnd best {best('rnd')} native {nat_tup} "
              f"jobs {n_jobs} {rec['seconds']:.0f}s sigma {rec['sigma']:.3f}", flush=True)

    if "locked" in state:
        return
    gens = state["generations"]
    noms = []
    for arm, key in (("cma", "cma_x"), ("rnd", "rnd_x")):
        pool = []
        for g in gens:
            for i, c in enumerate(g[arm]):
                pool.append((tuple(c["improvement"]), g["generation"], c["id"], g[key][i]))
        pool.sort(key=lambda t: (tuple(-v for v in t[0]), t[1], t[2]))
        k = 3 if arm == "cma" else 4
        for imp, gnum, cid, x in pool[:k]:
            noms.append((f"nom-{arm}-g{gnum:02d}-{cid}", arm, x))
        if arm == "cma":
            noms.append(("nom-cma-mean", "cma", gens[-1]["cma_mean"]))
    members = [(mid, adapter, {adapter: np.asarray(x)}, False) for mid, _, x in noms]
    res, _ = evaluate(q, members, select_ids, cases, natives, f"{args.tag}-{adapter}-r{args.repeat}{args.variant}-sel", log)
    table = []
    for (mid, arm, x) in noms:
        tup, parts = score(res[mid], natives)
        table.append({"id": mid, "arm": arm, "score": tup, "x": list(map(float, x)), **parts})
    locked = {}
    for arm in ("cma", "rnd"):
        rows = sorted([r for r in table if r["arm"] == arm], key=lambda r: (tuple(-v for v in r["score"]), r["id"]))
        locked[arm] = rows[0]
    state.update(selection=table, locked=locked, native_selection=native_score(select_ids, natives)[0])
    write(state_p, state)
    print(f"[{adapter} r{args.repeat}] locked cma {locked['cma']['id']} {locked['cma']['score']} | "
          f"rnd {locked['rnd']['id']} {locked['rnd']['score']} | native {state['native_selection']}", flush=True)


def untrained(adapter, k, seed, version=1):
    n = core.n_genes(adapter, version)
    rng = np.random.default_rng(seed + (0 if adapter == "steer" else 1) + (10 if version == 2 else 0))
    return [rng.uniform(-1.0, 1.0, n) for _ in range(k)]


def near_zero_steer():
    """Control: trigger always open (bias +3), authority 0.001 steps, along factor 1 (v = 0), u = 0."""
    g = dict.fromkeys(core.STEER_NAMES, 0.0)
    g["trig_bias"], g["release"], g["authority"] = 3.0, 2.0, 0.001
    x = []
    for name, lo, hi in zip(core.STEER_NAMES, core.STEER_LO, core.STEER_HI):
        x.append(2.0 * (g[name] - lo) / (hi - lo) - 1.0)
    return np.clip(np.array(x), -1, 1)


def mode_test(args, q):
    envs = args.envs.split(",")
    cases = load_panels(args.panels, envs)
    natives = load_natives(args.out, envs)
    test_ids = sorted(c for c, v in cases.items() if v["split"] == "test")
    locked = {}
    for p in Path(args.out).glob("*-r*/state.json"):
        s = read(p)
        # run 1 states live in <adapter>-r<k>/, run 2 in <adapter>-r<k><variant>/ (e.g. steer-r1-v2)
        if "locked" in s and p.parent.name == f"{s['adapter']}-r{s['repeat']}{args.variant}":
            locked[(s["adapter"], s["repeat"])] = s["locked"]
    pre = "v2-" if args.version == 2 else ""
    members = [("native", "native", {}, False)]
    for adapter in ("steer", "choose"):
        for (a, r), lk in sorted(locked.items()):
            if a != adapter:
                continue
            for arm in ("cma", "rnd"):
                members.append((f"{pre}{adapter}-{arm}-r{r}", adapter, {adapter: np.asarray(lk[arm]["x"])}, False))
            if adapter == "steer" and args.version == 1:
                members.append((f"steer-cma-r{r}-veto", "steer", {"steer": np.asarray(lk["cma"]["x"])}, True))
        for i, x in enumerate(untrained(adapter, 3 if args.version == 1 else 2, args.seed + 99, args.version)):
            members.append((f"{pre}{adapter}-untrained{i}", adapter, {adapter: x}, False))
    for r in sorted({r for (a, r) in locked}):
        if ("steer", r) in locked and ("choose", r) in locked:
            for arm in ("cma", "rnd"):
                members.append((f"{pre}stack-{arm}-r{r}", "stack", {"steer": np.asarray(locked[("steer", r)][arm]["x"]),
                                                               "choose": np.asarray(locked[("choose", r)][arm]["x"])}, False))
    if args.version == 1:
        members.append(("steer-nearzero", "steer", {"steer": near_zero_steer()}, False))
    if args.only:
        keep = set(args.only.split(","))
        members = [m for m in members if any(m[0].startswith(k) for k in keep)]
    print(f"test: {len(members)} arms x {len(test_ids)} cases on {envs}", flush=True)
    log = (Path(args.out) / f"episodes-test-{args.tag}.jsonl").open("a")
    res, n_jobs = evaluate(q, members, test_ids, cases, natives, f"{args.tag}-test", log)
    table = []
    for mid, adapter, xs, veto in members:
        for env in envs:
            items = [(o, n) for (o, n), cid in zip(res[mid], test_ids) if cases[cid]["env"] == env]
            for level in (1, 2, 3, 4):
                lv = [(o, n) for (o, n), cid in zip(res[mid], test_ids) if cases[cid]["env"] == env and cases[cid]["level"] == level]
                if lv:
                    tup, parts = score(lv, natives)
                    table.append({"arm": mid, "env": env, "level": level, "n": len(lv), **parts})
            tup, parts = score(items, natives)
            table.append({"arm": mid, "env": env, "level": "all", "n": len(items), **parts})
    write(Path(args.out) / f"test-{args.tag}.json", {"arms": [m[0] for m in members], "table": table, "jobs": n_jobs,
                                                     "per_case": {mid: [{k: o.get(k) for k in ("case_id", "success", "violation", "safe_completion", "min_clear_steps", "final_goal_steps", "steer_open", "choose_open", "free")} for o, _ in res[mid]] for mid in res}})
    print(f"test done: {n_jobs} jobs run", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("natives", "search", "test"))
    ap.add_argument("--queue", required=True)
    ap.add_argument("--panels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--envs", default="tworoom,pusht,basketball")
    ap.add_argument("--splits", default="train,select,test")
    ap.add_argument("--adapter", choices=("steer", "choose"))
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--pop", type=int, default=12)
    ap.add_argument("--sigma0", type=float, default=0.6)  # 0.3 of the [-1, 1] range
    ap.add_argument("--min-gens", type=int, default=12)
    ap.add_argument("--max-gens", type=int, default=24)
    ap.add_argument("--deadline", type=float, default=float("inf"), help="UTC epoch: no new generation after it (once min-gens is met)")
    ap.add_argument("--seed", type=int, default=20261109)
    ap.add_argument("--tag", default="g1")
    ap.add_argument("--only", default="")
    ap.add_argument("--train-envs", default=",".join(TRAIN_ENVS), help="search: training environments (follow-up: one env)")
    ap.add_argument("--all-cases", action="store_true", help="search: every training case each generation (follow-up: less noise)")
    ap.add_argument("--variant", default="", help="search: suffix for the output dir and job ids, e.g. -all or -tworoom")
    ap.add_argument("--version", type=int, default=1, choices=(1, 2), help="adapter version (2 = run 2: built-in direction)")
    ap.add_argument("--fixed", default="", help="search: fixed cases per env for levels 1,2,3 every generation, e.g. 1,2,1")
    args = ap.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    q = Queues(args.queue)
    {"natives": mode_natives, "search": mode_search, "test": mode_test}[args.mode](args, q)


if __name__ == "__main__":
    main()
