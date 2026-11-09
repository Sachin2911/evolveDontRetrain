"""Graded study: TD-MPC2 basketball file-queue worker (one process per GPU).

    python -m graded.td_worker --device N --queue <root>/basketball

Claims <queue>/jobs/<job_id>.json by rename, writes <queue>/results/<job_id>.json (graded/INTERFACES.md),
stops when <queue>/STOP exists or after --idle-exit seconds without work. Errors are recorded, never rerun.
"""

import argparse
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np


def write(path, obj):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj))
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", required=True, help="GPU index (N or cuda:N)")
    ap.add_argument("--queue", required=True)
    ap.add_argument("--idle-exit", type=float, default=36000.0)
    args = ap.parse_args()
    dev = str(args.device).split(":")[-1]
    tag = f"gpu{dev}-{os.getpid()}"
    queue = Path(args.queue)
    jobs, results = queue / "jobs", queue / "results"
    jobs.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)

    from graded import td_plugin

    td_plugin.init(dev)
    write(queue / f"ready-{tag}.json", {"device": dev, "pid": os.getpid(), "epoch": time.time()})
    print(json.dumps({"ready": tag}), flush=True)
    rng = np.random.default_rng()
    last = time.monotonic()
    while not (queue / "STOP").exists():
        pending = sorted(jobs.glob("*.json"))
        if not pending:
            if time.monotonic() - last > args.idle_exit:
                break
            time.sleep(0.5)
            continue
        path = pending[rng.integers(len(pending))]
        claimed = path.with_suffix(f".{tag}.claimed")
        try:
            path.rename(claimed)
        except FileNotFoundError:
            continue
        last = time.monotonic()
        try:
            job = json.loads(claimed.read_text())
        except Exception as error:  # half-written job file: put it back and retry later
            claimed.rename(path)
            print(json.dumps({"skip": path.name, "error": repr(error)}), flush=True)
            time.sleep(0.2)
            continue
        try:
            out = td_plugin.run(job)
        except Exception as error:
            case = job.get("case") or {}
            out = {"job_id": job.get("job_id"), "env": "basketball", "case_id": case.get("id"), "kind": job.get("kind"),
                   "completed": False, "error": repr(error), "traceback": traceback.format_exc()[-4000:]}
            print(json.dumps({"job": job.get("job_id"), "error": repr(error)}), flush=True)
        out["worker"] = tag
        write(results / (str(job.get("job_id")) + ".json"), out)
        try:
            claimed.unlink()
        except FileNotFoundError:
            pass
        if out.get("completed"):
            print(json.dumps({k: out.get(k) for k in ("job_id", "kind", "success", "violation", "min_clear_steps",
                                                      "steer_open", "choose_open", "checks_max", "seconds")}), flush=True)


if __name__ == "__main__":
    main()
