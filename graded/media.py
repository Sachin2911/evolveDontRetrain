"""Recorded reruns for the paper's trajectory figures (run on a host's production queues).

  python -m graded.media --queue Q --panels P --out OUT --tag m1 --cases c1,c2 --arms native,choose-cma-r1,v2-steer-cma-r1

Each (arm, case) runs once more with record = True. Every planner here is deterministic per case, so the rerun repeats
the test episode (checked: the outcome's parity_sha256 must equal the test's, printed per job). AP workers save
positions and frames to <queue>/<env>/media/<job_id>.npz and put positions in the result; the TD worker returns a
`trace` with the gripper and ball paths. Results land in <queue>/<env>/results/<job_id>.json as usual.

Arm names follow graded.search test mode: native | [v2-]<steer|choose|stack>-<cma|rnd>-r<k> | [v2-]<adapter>-untrained<i>.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from graded.search import Queues, load_panels, load_natives, make_job, read, untrained


def genotypes(out, arm, seed):
    """Arm name -> (adapter kind, xs dict)."""
    if arm == "native":
        return "native", {}
    version = 2 if arm.startswith("v2-") else 1
    name = arm[3:] if version == 2 else arm
    suffix = "-v2" if version == 2 else ""
    adapter = name.split("-")[0]
    if "untrained" in name:
        i = int(name.split("untrained")[1])
        return adapter, {adapter: untrained(adapter, i + 1, seed + 99, version)[i]}
    _, method, rep = name.split("-")
    r = int(rep[1:])
    xs = {}
    for a in (("steer", "choose") if adapter == "stack" else (adapter,)):
        s = read(Path(out) / f"{a}-r{r}{suffix}" / "state.json")
        xs[a] = np.asarray(s["locked"][method]["x"])
    return adapter, xs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", required=True)
    ap.add_argument("--panels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="m1")
    ap.add_argument("--cases", required=True)
    ap.add_argument("--arms", required=True)
    ap.add_argument("--seed", type=int, default=20261109)
    args = ap.parse_args()
    case_ids = args.cases.split(",")
    envs = sorted({c.split("-")[0] for c in case_ids})
    cases = load_panels(args.panels, envs)
    natives = load_natives(args.out, envs)
    tests = {}
    for p in Path(args.out).glob("test-*.json"):
        for arm, rows in read(p)["per_case"].items():
            tests.setdefault(arm, {}).update({r["case_id"]: r for r in rows})
    q = Queues(args.queue)
    jobs, meta = [], []
    for arm in args.arms.split(","):
        adapter, xs = genotypes(args.out, arm, args.seed)
        for cid in case_ids:
            if adapter == "native":
                job = {"job_id": f"{args.tag}-rec-native-{cid}", "env": cases[cid]["env"], "kind": "native",
                       "case": cases[cid], "log_senses": False}
            else:
                job = make_job(f"{args.tag}-rec-{arm}-{cid}", cases[cid], adapter, xs)
            job["record"] = True
            q.submit(job)
            jobs.append(job)
            meta.append((arm, cid))
    print(f"media: {len(jobs)} recorded jobs", flush=True)
    res = q.wait(jobs)
    summary = []
    for job, (arm, cid) in zip(jobs, meta):
        o = res[job["job_id"]]
        ref = natives[cid] if arm == "native" else tests.get(arm, {}).get(cid, {})
        same = (o.get("safe_completion"), o.get("violation"), o.get("success")) == \
               (ref.get("safe_completion"), ref.get("violation"), ref.get("success")) if ref else None
        if arm == "native":
            same = o.get("parity_sha256") == natives[cid].get("parity_sha256")
        summary.append({"arm": arm, "case": cid, "job": job["job_id"], "same_as_test": same,
                        "safe": o.get("safe_completion"), "violation": o.get("violation"), "success": o.get("success"),
                        "min_clear_steps": o.get("min_clear_steps"), "media": o.get("media"), "error": o.get("error")})
        print(json.dumps(summary[-1]), flush=True)
    (Path(args.out) / f"media-{args.tag}.json").write_text(json.dumps(summary, indent=1))
    print("MEDIA DONE", flush=True)


if __name__ == "__main__":
    main()
