"""Two-hour pilot: the TwoRoom adapter recipe on Block Plucking (SafeBlock/Pilot.md).

    nohup /venv/main/bin/python SafeBlock/experiments/pilot_search.py --phase all \
        > outputs/pilot-20261101/driver.log 2>&1 &

Phases in order: smoke, cma, evalA, random, evalB. Every phase is resumable (an
episode with a summary.json is never rerun). The smoke phase stops everything on a
parity failure. One simulator process per episode, two lanes.
"""

import argparse
import json
import pickle
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from probe_common import ROOT, atomic_json, project_path, utc  # noqa: E402

CONFIG = ROOT / "SafeBlock/configs/pilot-20261101.json"
FEATURES = ["retrieval_distance", "imagined_cost", "mean_arm", "closing_fraction", "progress"]
PHASES = ["smoke", "cma", "evalA", "random", "evalB"]


def status(root, **fields):
    atomic_json(root / "status.json", {"utc": utc(), **fields})


def episode_fitness(summary, weights):
    """Per-episode lexicographic fitness; a missing episode counts as violated, no success."""
    if summary is None:
        parts = {"sustained": 0, "success": 0, "violated": 1, "missing": 1, "controls": None}
    else:
        parts = {
            "sustained": int(bool(summary.get("sustained_safe_success"))),
            "success": int(bool(summary.get("safe_success"))),
            "violated": int(bool(summary.get("violation")) or bool(summary.get("post_success_violation"))),
            "missing": 0,
            "controls": summary.get("controls"),
        }
    value = weights["sustained"] * parts["sustained"] + weights["success"] * parts["success"] - weights["violated"] * parts["violated"]
    return float(value), parts


def write_genotype(path, params, source):
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(path, {"params": [float(v) for v in params], "features": FEATURES, "source": source, "utc": utc()})
    return path


def draw(seed, scale):
    import adapters

    return adapters.generic_genotype("selection", seed, scale)


def rows(attempt):
    return [json.loads(line) for line in (attempt / "steps.jsonl").read_text().splitlines() if line.strip()]


def find_attempt(job_root):
    for attempt in sorted(job_root.glob("attempt-*")):
        if (attempt / "summary.json").exists():
            return attempt
    return None


class Lanes:
    """A fixed number of concurrent worker processes, one episode each."""

    def __init__(self, cfg, root):
        self.cfg, self.root = cfg, root
        self.lock = threading.Lock()
        self.done = 0

    def command(self, job, attempt):
        s = self.cfg["system"]
        cmd = [
            sys.executable, str(ROOT / "SafeBlock/experiments/probe_worker.py"),
            "--task", s["task"], "--condition", s["condition"], "--seed", str(job["seed"]),
            "--output", str(attempt), "--config", str(ROOT / s["planner_config"]),
            "--library", str(project_path(s["library"])), "--freeze", s["freeze"],
        ]
        if job["adapter"] != "none":
            cmd += ["--adapter", job["adapter"], "--genotype-json", str(job["genotype_json"])]
        return cmd

    def one(self, phase, job):
        job_root = self.root / phase / job["id"]
        done = find_attempt(job_root)
        if done is not None:
            return job["id"], json.loads((done / "summary.json").read_text())
        for _ in range(self.cfg["execution"]["attempts_per_episode"]):
            attempts = sorted(job_root.glob("attempt-*"))
            attempt = job_root / f"attempt-{len(attempts) + 1:02d}"
            attempt.mkdir(parents=True)
            started = time.perf_counter()
            entry = {"job": job, "attempt": str(attempt.relative_to(ROOT)), "started_utc": utc()}
            with (attempt / "worker.log").open("w") as log:
                try:
                    completed = subprocess.run(
                        self.command(job, attempt), stdout=log, stderr=subprocess.STDOUT, cwd=ROOT,
                        timeout=self.cfg["execution"]["episode_timeout_s"],
                    )
                    entry["exit_code"] = completed.returncode
                except subprocess.TimeoutExpired:
                    entry["exit_code"] = "timeout"
            entry["seconds"] = time.perf_counter() - started
            entry["finished_utc"] = utc()
            summary = attempt / "summary.json"
            result = json.loads(summary.read_text()) if summary.exists() else None
            if result is not None:
                keys = ("outcome", "safe_success", "sustained_safe_success", "violation", "post_success_violation", "steps", "controls", "adapter_choice_changes", "adapter_blocks_changed")
                entry["result"] = {k: result.get(k) for k in keys}
            with self.lock:
                with (self.root / phase / "ledger.jsonl").open("a") as stream:
                    stream.write(json.dumps(entry) + "\n")
                self.done += 1
            if result is not None:
                return job["id"], result
        return job["id"], None

    def run(self, phase, jobs):
        (self.root / phase).mkdir(parents=True, exist_ok=True)
        with ThreadPoolExecutor(max_workers=self.cfg["execution"]["lanes"]) as pool:
            return dict(pool.map(lambda j: self.one(phase, j), jobs))


# ---------------------------------------------------------------- phases


def phase_smoke(cfg, lanes, root):
    """Zero weights through --genotype-json must reproduce the Stage 1 bare episodes bitwise."""
    zero = write_genotype(root / "genotypes" / "zero.json", [0.0] * 5, {"kind": "zero"})
    jobs = [
        {"id": f"smoke-zero-{s}", "phase": "smoke", "seed": s, "adapter": "selection", "genotype_json": str(zero)}
        for s in cfg["smoke"]["starts"]
    ]
    results = lanes.run("smoke", jobs)
    report = {"utc": utc(), "pairs": [], "passed": True}
    for job in jobs:
        ours = find_attempt(root / "smoke" / job["id"])
        ref = find_attempt(project_path(cfg["smoke"]["reference_root"]) / f"S1C-{job['seed']}-bare")
        if ours is None or ref is None:
            report["pairs"].append({"seed": job["seed"], "error": "missing episode", "ours": str(ours), "reference": str(ref)})
            report["passed"] = False
            continue
        ra, rb = rows(ref), rows(ours)
        n = min(len(ra), len(rb))
        same_action = sum(ra[i]["executed_normalized_action"] == rb[i]["executed_normalized_action"] for i in range(n))
        same_camera = sum(ra[i]["camera_sha256"] == rb[i]["camera_sha256"] for i in range(n))
        sr, so = json.loads((ref / "summary.json").read_text()), results[job["id"]]
        outcome_keys = ("safe_success", "sustained_safe_success", "violation", "post_success_violation")
        same_outcome = all(bool(sr.get(k)) == bool(so.get(k)) for k in outcome_keys) and sr.get("steps") == so.get("steps")
        changed = int(so.get("adapter_choice_changes") or 0) + int(so.get("adapter_blocks_changed") or 0)
        ok = same_outcome and len(ra) == len(rb) and same_action == n and same_camera == n and changed == 0
        report["pairs"].append({
            "seed": job["seed"], "controls_reference": len(ra), "controls_ours": len(rb),
            "same_action": same_action, "same_camera": same_camera, "same_outcome": same_outcome,
            "adapter_changes": changed, "passed": ok, "reference": str(ref.relative_to(ROOT)), "ours": str(ours.relative_to(ROOT)),
        })
        report["passed"] = report["passed"] and ok
    atomic_json(root / "smoke" / "parity.json", report)
    return report["passed"]


def make_optimizer(c):
    import cma
    from cma.sigma_adaptation import CMAAdaptSigmaCSA

    return cma.CMAEvolutionStrategy(
        np.zeros(c["dimension"]), c["sigma0"],
        dict(CMA_diagonal=True, AdaptSigma=CMAAdaptSigmaCSA, CMA_mirrors=0, popsize=c["population"],
             seed=c["seed"], verbose=-9, verb_log=0, verb_disp=0, maxiter=c["generations"]),
    )


def phase_cma(cfg, lanes, root):
    c, w = cfg["cma"], cfg["fitness"]
    cdir = root / "cma"
    cdir.mkdir(parents=True, exist_ok=True)
    state_path, trace_path = cdir / "state.pkl", cdir / "trace.json"
    if state_path.exists() and trace_path.exists():
        es = pickle.loads(state_path.read_bytes())
        trace = json.loads(trace_path.read_text())
    else:
        es = make_optimizer(c)
        trace = {"settings": c, "generations": []}
    for g in range(len(trace["generations"]) + 1, c["generations"] + 1):
        status(root, phase="cma", generation=g, episodes_done=lanes.done)
        asked_path = cdir / f"gen-{g:02d}-asked.json"
        candidates = [np.asarray(x, dtype=float) for x in es.ask()]
        if asked_path.exists():  # resume: reuse the candidates already drawn for this generation
            candidates = [np.asarray(x, dtype=float) for x in json.loads(asked_path.read_text())["candidates"]]
        else:
            atomic_json(asked_path, {"generation": g, "candidates": [x.tolist() for x in candidates], "sigma": float(es.sigma), "mean": es.mean.tolist(), "utc": utc()})
        starts = cfg["training_starts"][g - 1]
        jobs = []
        for i, x in enumerate(candidates):
            path = write_genotype(root / "genotypes" / f"cma-g{g:02d}-c{i}.json", x, {"kind": "cma", "generation": g, "candidate": i})
            for s in starts:
                jobs.append({"id": f"cma-g{g:02d}-c{i}-{s}", "phase": "cma", "seed": s, "adapter": "selection", "genotype_json": str(path), "generation": g, "candidate": i})
        results = lanes.run("cma", jobs)
        scores, episodes = [], []
        for i, x in enumerate(candidates):
            total, parts = 0.0, []
            for s in starts:
                value, p = episode_fitness(results.get(f"cma-g{g:02d}-c{i}-{s}"), w)
                total += value
                parts.append({"seed": s, "fitness": value, **p})
            scores.append(total)
            episodes.append(parts)
        es.tell(candidates, [-v for v in scores])
        record = {
            "generation": g, "starts": starts, "candidates": [x.tolist() for x in candidates], "fitness": scores,
            "episodes": episodes, "best": max(scores), "mean_fitness": float(np.mean(scores)),
            "mean_after": es.mean.tolist(), "sigma_after": float(es.sigma), "utc": utc(),
        }
        try:
            record["coordinate_std_after"] = np.asarray(es.stds).tolist()
        except Exception:  # noqa: BLE001
            pass
        trace["generations"].append(record)
        state_path.write_bytes(pickle.dumps(es))
        atomic_json(trace_path, trace)
    nominee = es.mean.tolist()
    write_genotype(root / "genotypes" / "cma-nominee.json", nominee, {"kind": "cma_nominee", "rule": "final distribution mean after the last generation", "generations": c["generations"]})
    status(root, phase="cma", generation="done", episodes_done=lanes.done, nominee=nominee)


def phase_random(cfg, lanes, root):
    r, w = cfg["random_search"], cfg["fitness"]
    starts = [s for pair in cfg["training_starts"][: r["generations_of_starts"]] for s in pair]
    status(root, phase="random", episodes_done=lanes.done)
    jobs, files = [], {}
    for seed in range(r["first_seed"], r["first_seed"] + r["draws"]):
        path = write_genotype(root / "genotypes" / f"random-{seed}.json", draw(seed, r["scale"]), {"kind": "random", "generator_seed": seed, "scale": r["scale"]})
        files[seed] = path
        for s in starts:
            jobs.append({"id": f"random-{seed}-{s}", "phase": "random", "seed": s, "adapter": "selection", "genotype_json": str(path), "generator_seed": seed})
    results = lanes.run("random", jobs)
    table = []
    for seed, path in files.items():
        total, parts = 0.0, []
        for s in starts:
            value, p = episode_fitness(results.get(f"random-{seed}-{s}"), w)
            total += value
            parts.append({"seed": s, "fitness": value, **p})
        table.append({"generator_seed": seed, "fitness": total, "episodes": parts, "params": json.loads(path.read_text())["params"]})
    best = max(table, key=lambda row: (row["fitness"], -row["generator_seed"]))
    write_genotype(root / "genotypes" / "random-nominee.json", best["params"], {"kind": "random_nominee", "generator_seed": best["generator_seed"], "fitness": best["fitness"], "rule": "highest summed fitness on the eight training starts; ties to the lower seed"})
    atomic_json(root / "random" / "search.json", {"starts": starts, "draws": table, "nominee": best["generator_seed"], "utc": utc()})
    status(root, phase="random", episodes_done=lanes.done, nominee=best["generator_seed"])


def phase_eval(cfg, lanes, root, arms, name):
    panel = cfg["evaluation_panel"]
    status(root, phase=name, episodes_done=lanes.done)
    jobs = []
    for arm in arms:
        if arm == "native":
            spec = {"adapter": "none", "genotype_json": None}
        else:
            path = root / "genotypes" / {"untrained": "untrained.json", "cma": "cma-nominee.json", "random": "random-nominee.json"}[arm]
            if arm == "untrained" and not path.exists():
                write_genotype(path, draw(cfg["untrained"]["seed"], cfg["untrained"]["scale"]), {"kind": "untrained", "generator_seed": cfg["untrained"]["seed"], "scale": cfg["untrained"]["scale"]})
            if not path.exists():
                raise SystemExit(f"{path.name} is missing: run its search phase first")
            spec = {"adapter": "selection", "genotype_json": str(path)}
        for s in panel:
            jobs.append({"id": f"eval-{arm}-{s}", "phase": "eval", "seed": s, "arm": arm, **spec})
    results = lanes.run("eval", jobs)
    receipt = {"utc": utc(), "arms": {}}
    for arm in arms:
        counts = {"episodes": 0, "sustained": 0, "success": 0, "violated": 0, "missing": 0}
        for s in panel:
            _, p = episode_fitness(results.get(f"eval-{arm}-{s}"), cfg["fitness"])
            counts["episodes"] += 1
            for k in ("sustained", "success", "violated", "missing"):
                counts[k] += p[k]
        receipt["arms"][arm] = counts
    atomic_json(root / "eval" / f"{name}.json", receipt)
    status(root, phase=name, episodes_done=lanes.done, result=receipt["arms"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["all"] + PHASES, default="all")
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--output", type=Path, default=project_path("outputs/pilot-20261101"))
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    root = project_path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    atomic_json(root / f"driver-start-{int(time.time())}.json", {"utc": utc(), "phase": args.phase, "config": str(args.config), "python": sys.executable})
    lanes = Lanes(cfg, root)
    phases = PHASES if args.phase == "all" else [args.phase]
    for phase in phases:
        tick = time.perf_counter()
        if phase == "smoke":
            if not phase_smoke(cfg, lanes, root):
                status(root, phase="smoke", failed=True, episodes_done=lanes.done)
                print("PARITY SMOKE FAILED; stopping", flush=True)
                sys.exit(2)
        elif phase == "cma":
            phase_cma(cfg, lanes, root)
        elif phase == "evalA":
            phase_eval(cfg, lanes, root, ["native", "untrained", "cma"], "evalA")
        elif phase == "random":
            phase_random(cfg, lanes, root)
        elif phase == "evalB":
            phase_eval(cfg, lanes, root, ["random"], "evalB")
        atomic_json(root / f"phase-{phase}.json", {"utc": utc(), "phase": phase, "seconds": time.perf_counter() - tick, "episodes_done_total": lanes.done})
        print(json.dumps({"phase": phase, "done": True, "seconds": round(time.perf_counter() - tick), "episodes": lanes.done}), flush=True)
    status(root, phase="complete" if args.phase == "all" else args.phase, episodes_done=lanes.done)


if __name__ == "__main__":
    main()
