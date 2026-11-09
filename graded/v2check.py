"""Run 2 smoke check for the v2 adapters, on a host's production queues.

  python -m graded.v2check --queue Q --panels P --out OUT --envs tworoom,pusht,basketball,reacher --tag v2c1

Per env, one test case where native completes but violates (level 2):
  - x = 0 Steer v2 and Choose v2, run for real (not as free outcomes): parity_sha256 must equal native's;
  - trigger forced open (trig_bias = +3, other genes at the box centre) for Steer v2, Choose v2 and the stack:
    must complete without an error. Prints outcomes and gate counts.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from graded import core
from graded.search import Queues, load_panels, load_natives, make_job


def forced(adapter):
    n = core.n_genes(adapter, 2)
    names = core.BOXES[adapter + "2"][0]
    x = np.zeros(n)
    x[names.index("trig_bias")] = 1.0
    if adapter == "steer":
        x[names.index("release")] = 1.0
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", required=True)
    ap.add_argument("--panels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--envs", default="tworoom,pusht,basketball,reacher")
    ap.add_argument("--tag", default="v2c1")
    args = ap.parse_args()
    envs = args.envs.split(",")
    cases = load_panels(args.panels, envs)
    natives = load_natives(args.out, envs)
    q = Queues(args.queue)
    jobs, meta = [], []
    for env in envs:
        pick = sorted(c for c, v in cases.items() if v["env"] == env and v["split"] == "test" and v["level"] == 2
                      and natives[c].get("success") and natives[c].get("violation"))[:1]
        for cid in pick:
            specs = [("zero-steer", "steer", {"steer": np.zeros(core.N_STEER2)}),
                     ("zero-choose", "choose", {"choose": np.zeros(core.N_CHOOSE2)}),
                     ("open-steer", "steer", {"steer": forced("steer")}),
                     ("open-choose", "choose", {"choose": forced("choose")}),
                     ("open-stack", "stack", {"steer": forced("steer"), "choose": forced("choose")})]
            for name, adapter, xs in specs:
                job = make_job(f"{args.tag}-{name}-{cid}", cases[cid], adapter, xs)
                q.submit(job)
                jobs.append(job)
                meta.append((env, cid, name))
    print(f"v2check: {len(jobs)} jobs", flush=True)
    res = q.wait(jobs)
    bad = 0
    for job, (env, cid, name) in zip(jobs, meta):
        o, nat = res[job["job_id"]], natives[cid]
        line = {k: o.get(k) for k in ("completed", "success", "violation", "safe_completion", "min_clear_steps",
                                       "steer_open", "choose_open", "error")}
        if name.startswith("zero"):
            same = o.get("parity_sha256") == nat.get("parity_sha256")
            line["parity_same_as_native"] = same
            bad += not same
        else:
            bad += not o.get("completed")
        print(env, name, json.dumps(line), flush=True)
    print("V2CHECK", "PASS" if bad == 0 else f"FAIL {bad}", flush=True)


if __name__ == "__main__":
    main()
