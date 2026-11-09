"""residualRecipe POC driver: native baseline, parity, CMA-ES vs random search, held-out nominees.

Runs on the rented host's CPU and talks to GPU workers through a job directory.
Phases:
  0  cases          build frozen hazard cases from the qd_adapters splits
  1  native/parity  native on train + held-out cases; parity member on two cases
  2  search         G generations, each: `pop` CMA-ES candidates + `pop` random-search
                    candidates, all on the same training cases (common random numbers)
  3  nominees       top-k CMA, top-k random, k untrained draws on held-out cases
Fitness is the order (safe completions, completions, -violations, mean min clearance).
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np

from qd_adapters.io import read, serial, write
from residual.hook import N_GENES, random_genotype


class Queue:
    def __init__(self, root):
        self.root = Path(root)
        self.jobs, self.results = self.root / "jobs", self.root / "results"
        self.jobs.mkdir(parents=True, exist_ok=True)
        self.results.mkdir(parents=True, exist_ok=True)
        self.log = (self.root / "episodes.jsonl").open("a")

    def submit(self, job):
        write(self.jobs / (job["job_id"] + ".json"), job)

    def wait(self, job_ids, poll=2.0):
        out = {}
        while len(out) < len(job_ids):
            for job_id in job_ids:
                if job_id in out:
                    continue
                path = self.results / (job_id + ".json")
                if path.exists():
                    try:
                        out[job_id] = read(path)
                    except json.JSONDecodeError:
                        continue
                    self.log.write(json.dumps(serial(out[job_id]), sort_keys=True) + "\n")
                    self.log.flush()
            if len(out) < len(job_ids):
                time.sleep(poll)
        return [out[j] for j in job_ids]


def fitness(outcomes, scale):
    ok = [o for o in outcomes if o.get("completed")]
    safe = sum(bool(o["safe_completion"]) for o in ok)
    complete = sum(bool(o["success"]) for o in ok)
    violations = sum(bool(o["violation"]) for o in ok)
    clear = float(np.mean([o["swept_min_clearance"] for o in ok]) / scale) if ok else -9.0
    return (safe, complete, -violations, round(clear, 4)), len(ok)


def rank_fitness(tuples):
    order = sorted(range(len(tuples)), key=lambda i: (tuple(-v for v in tuples[i]), i))
    ranks = [0] * len(tuples)
    for rank, index in enumerate(order):
        ranks[index] = rank
    return ranks


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("task")
    ap.add_argument("--queue", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--train-splits", default="architecture_train,pool_train,calibration")
    ap.add_argument("--heldout-splits", default="deployment")
    ap.add_argument("--n-train", type=int, default=8)
    ap.add_argument("--n-heldout", type=int, default=32)
    ap.add_argument("--pop", type=int, default=8)
    ap.add_argument("--generations", type=int, default=8)
    ap.add_argument("--sigma", type=float, default=0.5)
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--seed", type=int, default=20261108)
    ap.add_argument("--phases", default="1,2,3")
    ap.add_argument("--scale", type=float, default=None, help="physical scale for clearance units")
    ap.add_argument("--tag", default="", help="prefix for job ids, so repeats never reuse an identity")
    ap.add_argument("--enrich", action="store_true", help="choose training cases where native is not safe")
    ap.add_argument("--scalar", action="store_true", help="rank CMA by the weighted sum S+C-V instead of the order (device-1 control)")
    ap.add_argument("--n-media", type=int, default=4, help="held-out cases to record in phase 4")
    args = ap.parse_args()
    from qd_adapters.geometry import SCALES

    scale = args.scale or SCALES[args.task]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    q = Queue(args.queue)
    phases = set(int(p) for p in args.phases.split(","))
    rng = np.random.default_rng(args.seed)
    state_path = out / "state.json"
    state = read(state_path) if state_path.exists() else {}

    # ---- phase 0: cases
    if "train_cases" not in state:
        jid = f"{args.tag}{args.task}-cases"
        q.submit({"job_id": jid, "kind": "cases", "splits": args.train_splits.split(",") + args.heldout_splits.split(",")})
        cases = q.wait([jid])[0]["cases"]
        hazard = [c for c in cases if c["condition"] != "clean"]
        train = [c for c in hazard if c["row"]["split"] in args.train_splits.split(",")]
        held = [c for c in hazard if c["row"]["split"] in args.heldout_splits.split(",")]
        # Deterministic selection of the panel sizes.
        prng = np.random.default_rng(args.seed + 1)
        if args.enrich:
            state["candidate_train_cases"] = train
        else:
            train = [train[i] for i in sorted(prng.choice(len(train), min(args.n_train, len(train)), replace=False))]
        held = [held[i] for i in sorted(prng.choice(len(held), min(args.n_heldout, len(held)), replace=False))]
        state.update(train_cases=train, heldout_cases=held, all_hazard_cases=len(hazard))
        write(state_path, state)
    train, held = state["train_cases"], state["heldout_cases"]
    print(f"[{args.task}] train cases {len(train)}, held-out cases {len(held)}", flush=True)

    def submit_member(phase, member, cases, native_scale):
        ids = []
        for case in cases:
            jid = f"{args.tag}{args.task}-{phase}-{member['id']}-{case['id']}"
            q.submit({"job_id": jid, "phase": phase, "case": case, "member": member, "native_scale": native_scale})
            ids.append(jid)
        return ids

    # ---- phase 1: native + parity
    if 1 in phases and "native_scale" not in state:
        native = {"id": "native", "kind": "native"}
        ids = submit_member("native", native, train + held, 1.0)
        n_train_jobs = len(train)
        zero = np.zeros(N_GENES).tolist()
        parity = {"id": "parity", "kind": "parity", "genes": zero}
        pids = submit_member("parity", parity, train[:2], 1.0)
        results = q.wait(ids + pids)
        nat, par = results[: len(ids)], results[len(ids):]
        native_scale = float(np.mean([r["mean_abs_action"] for r in nat if r.get("mean_abs_action")]))
        parity_ok = all(
            a["parity_sha256"] == b["parity_sha256"] and a["actions"] == b["actions"]
            for a, b in zip(nat[: len(pids)], par)
        )
        by_case = {r["case_id"]: [r["success"], r["violation"], r["safe_completion"]] for r in nat}
        if args.enrich:
            # Training panel = cases native does not solve safely; violated first, then unfinished.
            failing = [c for c in train if not by_case[c["id"]][2]]
            failing.sort(key=lambda c: (0 if by_case[c["id"]][1] else 1, c["id"]))
            state["enrichment"] = {"candidates": len(train), "native_unsafe": len(failing), "chosen": min(args.n_train, len(failing))}
            train = failing[: args.n_train]
            state["train_cases"] = train
        ft, n = fitness([r for r in nat if r["case_id"] in {c["id"] for c in train}], scale)
        fh, nh = fitness([r for r in nat if r["case_id"] in {c["id"] for c in held}], scale)
        state.update(
            native_scale=native_scale,
            parity_ok=parity_ok,
            native_train={"fitness": ft, "n": n},
            native_heldout={"fitness": fh, "n": nh},
            native_heldout_by_case=by_case,
            native_train_by_case={c["id"]: by_case[c["id"]] for c in train},
        )
        write(state_path, state)
        print(f"[{args.task}] native train {ft} ({n}); held-out {fh} ({nh}); scale |a| {native_scale:.3f}; parity {parity_ok}", flush=True)

    native_scale = state["native_scale"]
    train = state["train_cases"]

    # ---- phase 2: search
    if 2 in phases:
        import cma

        gens = state.setdefault("generations", [])
        es_seed = args.seed + 7
        es = cma.CMAEvolutionStrategy(
            np.zeros(N_GENES), args.sigma,
            {"popsize": args.pop, "CMA_diagonal": True, "seed": es_seed, "verbose": -9, "verb_log": 0,
             "verb_disp": 0, "tolfun": 0, "tolfunhist": 0, "tolflatfitness": 100000, "maxiter": 100000},
        )
        # Replay finished generations so the run can resume after an interruption. pycma needs an
        # ask before every tell, so each replayed generation asks (discarding the draw) then tells
        # the stored genes and ranks; with no generations left the strategy is not needed at all.
        for g in (gens if len(gens) < args.generations else []):
            es.ask()
            es.tell([np.asarray(x) for x in g["cma_genes"]], g["cma_ranks"])
        rrng = np.random.default_rng(args.seed + 11 + len(gens))
        best = {"cma": None, "random": None}
        for g in gens:
            for arm in ("cma", "random"):
                for cand in g[arm]:
                    if best[arm] is None or tuple(cand["fitness"]) > tuple(best[arm]["fitness"]):
                        best[arm] = cand
        for gen in range(len(gens), args.generations):
            cma_genes = es.ask()
            rand_genes = [rrng.normal(0.0, args.sigma, N_GENES) for _ in range(args.pop)]
            members = []
            for i, genes in enumerate(cma_genes):
                members.append(("cma", {"id": f"cma-g{gen:02d}-i{i:02d}", "kind": "residual", "genes": np.asarray(genes).tolist()}))
            for i, genes in enumerate(rand_genes):
                members.append(("random", {"id": f"rnd-g{gen:02d}-i{i:02d}", "kind": "residual", "genes": genes.tolist()}))
            ids = {m["id"]: submit_member("search", m, train, native_scale) for _, m in members}
            flat = [j for v in ids.values() for j in v]
            results = {r["job_id"]: r for r in q.wait(flat)}
            record = {"generation": gen, "cma": [], "random": [], "cma_genes": [np.asarray(x).tolist() for x in cma_genes]}
            for arm, m in members:
                outs = [results[j] for j in ids[m["id"]]]
                ft, n = fitness(outs, scale)
                cand = {"id": m["id"], "genes": m["genes"], "fitness": ft, "n": n,
                        "gate_open": int(sum(o.get("gate_open", 0) for o in outs)),
                        "vetoed": int(sum(o.get("vetoed", 0) for o in outs)),
                        "errors": int(sum(1 for o in outs if not o.get("completed")))}
                record[arm].append(cand)
                if best[arm] is None or ft > tuple(best[arm]["fitness"]):
                    best[arm] = cand
            if args.scalar:
                record["cma_ranks"] = rank_fitness([(c["fitness"][0] + c["fitness"][1] + c["fitness"][2], c["fitness"][3]) for c in record["cma"]])
                record["ranking"] = "scalar S+C-V"
            else:
                record["cma_ranks"] = rank_fitness([tuple(c["fitness"]) for c in record["cma"]])
            es.tell([np.asarray(x) for x in cma_genes], record["cma_ranks"])
            record["best_so_far"] = {arm: {"id": best[arm]["id"], "fitness": best[arm]["fitness"]} for arm in best}
            record["sigma"] = float(es.sigma)
            gens.append(record)
            write(state_path, state)
            print(
                f"[{args.task}] gen {gen}: cma best {max(tuple(c['fitness']) for c in record['cma'])} "
                f"random best {max(tuple(c['fitness']) for c in record['random'])} | best-so-far cma "
                f"{best['cma']['fitness']} random {best['random']['fitness']} | native {state['native_train']['fitness']}",
                flush=True,
            )

    # ---- phase 3: nominees on held-out
    if 3 in phases and "heldout" not in state:
        gens = state["generations"]
        cands = {"cma": [], "random": []}
        for g in gens:
            for arm in cands:
                cands[arm].extend(g[arm])
        nominees = []
        for arm in cands:
            ranked = sorted(cands[arm], key=lambda c: (tuple(-v for v in c["fitness"]), c["id"]))
            for c in ranked[: args.top]:
                nominees.append((arm, {"id": f"nom-{c['id']}", "kind": "residual", "genes": c["genes"], "train_fitness": c["fitness"]}))
        urng = np.random.default_rng(args.seed + 99)
        for i in range(args.top):
            nominees.append(("untrained", {"id": f"untrained-{i}", "kind": "residual", "genes": urng.normal(0.0, args.sigma, N_GENES).tolist()}))
        ids = {m["id"]: submit_member("heldout", m, held, native_scale) for _, m in nominees}
        flat = [j for v in ids.values() for j in v]
        results = {r["job_id"]: r for r in q.wait(flat)}
        native_by_case = state["native_heldout_by_case"]
        table = []
        for arm, m in nominees:
            outs = [results[j] for j in ids[m["id"]]]
            ft, n = fitness(outs, scale)
            paired = {"safe_gain": 0, "safe_loss": 0, "viol_added": 0, "viol_removed": 0}
            for o in outs:
                nat = native_by_case.get(o["case_id"])
                if not nat or not o.get("completed"):
                    continue
                paired["safe_gain"] += int(o["safe_completion"] and not nat[2])
                paired["safe_loss"] += int(nat[2] and not o["safe_completion"])
                paired["viol_added"] += int(o["violation"] and not nat[1])
                paired["viol_removed"] += int(nat[1] and not o["violation"])
            table.append({"arm": arm, "id": m["id"], "train_fitness": m.get("train_fitness"), "heldout_fitness": ft, "n": n,
                          "gate_open": int(sum(o.get("gate_open", 0) for o in outs)),
                          "vetoed": int(sum(o.get("vetoed", 0) for o in outs)),
                          "hook_decisions": int(sum(o.get("hook_decisions", 0) for o in outs)), **paired})
        state["heldout"] = {"native": state["native_heldout"], "nominees": table}
        write(state_path, state)
        print(f"[{args.task}] HELD-OUT native {state['native_heldout']['fitness']} n={state['native_heldout']['n']}", flush=True)
        for row in table:
            print(json.dumps(row | {"train_fitness": row["train_fitness"]}, default=str), flush=True)
    # ---- phase 4: media (record frames for native, top CMA, top random, untrained on violated held-out cases)
    if 4 in phases and "media" not in state and "heldout" in state:
        by_case = state["native_heldout_by_case"]
        violated = [c for c in held if by_case.get(c["id"], [0, 0, 0])[1]]
        cases = (violated + [c for c in held if c not in violated])[: args.n_media]
        noms = state["heldout"]["nominees"]
        pick = {}
        for arm in ("cma", "random", "untrained"):
            rows = [r for r in noms if r["arm"] == arm]
            if rows:
                pick[arm] = rows[0]["id"]
        genes = {}
        for g in state["generations"]:
            for arm in ("cma", "random"):
                for c in g[arm]:
                    genes["nom-" + c["id"]] = c["genes"]
        members = [{"id": "media-native", "kind": "native"}]
        for arm, nid in pick.items():
            if arm == "untrained":
                urng = np.random.default_rng(args.seed + 99)
                members.append({"id": "media-untrained-0", "kind": "residual", "genes": urng.normal(0.0, args.sigma, N_GENES).tolist()})
            else:
                members.append({"id": "media-" + nid, "kind": "residual", "genes": genes[nid]})
        media_dir = str(out / "media")
        ids = []
        for m in members:
            for case in cases:
                jid = f"{args.tag}{args.task}-media-{m['id']}-{case['id']}"
                q.submit({"job_id": jid, "phase": "media", "case": case, "member": m, "native_scale": native_scale, "record": True, "media_dir": media_dir})
                ids.append(jid)
        results = q.wait(ids)
        state["media"] = [{"job_id": r["job_id"], "member": r.get("member_id"), "case_id": r.get("case_id"), "success": r.get("success"), "violation": r.get("violation"), "file": r.get("media"), "gate_open": r.get("gate_open"), "vetoed": r.get("vetoed")} for r in results]
        write(state_path, state)
        print(f"[{args.task}] media recorded: {len(results)} episodes -> {media_dir}", flush=True)
    # ---- phase 5: hand-set structure control (device 2): c = 0.15, onset = 1.0, random weights
    if 5 in phases and "handset" not in state:
        from residual.hook import N_WEIGHTS

        hrng = np.random.default_rng(args.seed + 555)
        members = []
        for i in range(args.top):
            g = hrng.normal(0.0, args.sigma, N_GENES)
            g[N_WEIGHTS] = -1.253   # c = 0.15 of native magnitude
            g[N_WEIGHTS + 1] = -0.981  # onset = 1.0 hazard radii
            members.append({"id": f"handset-{i}", "kind": "residual", "genes": g.tolist()})
        ids = {m["id"]: submit_member("handset", m, held, native_scale) for m in members}
        flat = [j for v in ids.values() for j in v]
        results = {r["job_id"]: r for r in q.wait(flat)}
        native_by_case = state["native_heldout_by_case"]
        table = []
        for m in members:
            outs = [results[j] for j in ids[m["id"]]]
            ft, n = fitness(outs, scale)
            paired = {"safe_gain": 0, "safe_loss": 0, "viol_added": 0, "viol_removed": 0}
            for o in outs:
                nat = native_by_case.get(o["case_id"])
                if not nat or not o.get("completed"):
                    continue
                paired["safe_gain"] += int(o["safe_completion"] and not nat[2])
                paired["safe_loss"] += int(nat[2] and not o["safe_completion"])
                paired["viol_added"] += int(o["violation"] and not nat[1])
                paired["viol_removed"] += int(nat[1] and not o["violation"])
            table.append({"arm": "handset", "id": m["id"], "train_fitness": None, "heldout_fitness": ft, "n": n,
                          "gate_open": int(sum(o.get("gate_open", 0) for o in outs)), "vetoed": int(sum(o.get("vetoed", 0) for o in outs)),
                          "hook_decisions": int(sum(o.get("hook_decisions", 0) for o in outs)), **paired})
            print(json.dumps(table[-1]), flush=True)
        state["handset"] = table
        write(state_path, state)
    write(out / "summary.json", {k: v for k, v in state.items() if k not in ("train_cases", "heldout_cases", "candidate_train_cases")})


if __name__ == "__main__":
    main()
