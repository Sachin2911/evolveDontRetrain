"""Hung-job watchdog for AP workers (added 17:05 UTC after one Reacher worker hung on a lock).

  python -m graded.watchdog <queue_root> <env> [--limit 180]

A job claimed (renamed to <id>.<tag>.claimed) for longer than --limit seconds with no result is treated as hung:
  1. dump the worker's Python stack with py-spy to logs/hang-<id>.txt (evidence);
  2. kill that worker and start a fresh one on the same device (own LEWM_RUN_DIR);
  3. rerun the job under a fresh id (the ledger refuses reused ids) and, once it finishes, write its outcome as the
     ORIGINAL id's result, so the waiting driver continues. The outcome records `rerun_of_hang: <new id>`.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import time


def worker_pid(env, device):
    out = subprocess.run(["ps", "-eo", "pid,args"], stdout=subprocess.PIPE, text=True).stdout
    for line in out.splitlines():
        parts = line.split(None, 1)
        if len(parts) == 2 and parts[1].startswith(f"/venv/main/bin/python -m graded.ap_worker {env} --device {device} "):
            return int(parts[0])
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("env")
    ap.add_argument("--limit", type=float, default=180.0)
    args = ap.parse_args()
    q = Path(args.root) / args.env
    logs = Path(os.environ["WS"]) / "runs/graded/logs"
    pending = {}  # original id -> new id
    # Resume: reruns already submitted by an earlier watchdog (any "<id>-wd" job, claim or result).
    for f in list((q / "jobs").glob("*-wd*")) + list((q / "results").glob("*-wd.json")):
        nid = f.name.split(".")[0]
        orig = nid[: -len("-wd")]
        if not (q / "results" / (orig + ".json")).exists():
            pending[orig] = nid
    while True:
        now = time.time()
        for f in sorted((q / "jobs").glob("*.claimed")):
            name = f.name[: -len(".claimed")]
            jid, tag = name.rsplit(".", 1)
            if jid in pending or (q / "results" / (jid + ".json")).exists():
                continue
            if jid.endswith("-wd"):
                # run 2: a rerun that hangs too is deterministic; score it as a failure (the pending entry then copies
                # this result to the original id) and restart the worker.
                try:
                    age = now - f.stat().st_ctime
                except FileNotFoundError:
                    continue
                if age < args.limit:
                    continue
                device = "cuda:" + tag[len("cuda"):]
                pid = worker_pid(args.env, device)
                if pid and Path(f"/proc/{pid}").stat().st_ctime < f.stat().st_ctime:
                    os.kill(pid, 9)
                    time.sleep(3)
                    subprocess.run(["bash", "graded/run.sh", "ap", args.env, device], cwd=os.environ["SRC"],
                                   stdin=subprocess.DEVNULL)
                job = json.loads(f.read_text())
                fail = {"job_id": jid, "env": job["env"], "case_id": job["case"]["id"], "completed": False,
                        "error": "hung twice (original + watchdog rerun); scored as a failure by the watchdog"}
                tmp = q / "results" / (jid + ".json.tmp")
                tmp.write_text(json.dumps(fail))
                tmp.rename(q / "results" / (jid + ".json"))
                f.rename(f.with_suffix(".hung"))
                print(json.dumps({"hung_twice": jid, "age": round(age), "device": device, "pid": pid}), flush=True)
                continue
            try:
                age = now - f.stat().st_ctime
            except FileNotFoundError:
                continue
            if age < args.limit:
                continue
            device = "cuda:" + tag[len("cuda"):]
            pid = worker_pid(args.env, device)
            if pid and Path(f"/proc/{pid}").stat().st_ctime > f.stat().st_ctime:
                pid = None  # the live worker started after this claim: an orphan of a dead worker, not a hang
                orphan = True
            else:
                orphan = False
            note = logs / f"hang-{jid}.txt"
            if pid:
                dump = subprocess.run(["py-spy", "dump", "--pid", str(pid)], stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT, text=True, timeout=60).stdout
                note.write_text(f"job {jid} hung {age:.0f}s on {device} pid {pid}\n{dump}")
                os.kill(pid, 9)
                time.sleep(3)
            if not orphan:
                subprocess.run(["bash", "graded/run.sh", "ap", args.env, device], cwd=os.environ["SRC"],
                               stdin=subprocess.DEVNULL)
            job = json.loads(f.read_text())
            new = dict(job, job_id=job["job_id"] + "-wd")
            (q / "jobs" / (new["job_id"] + ".json")).write_text(json.dumps(new))
            pending[jid] = new["job_id"]
            f.rename(f.with_suffix(".hung"))
            print(json.dumps({"hung": jid, "age": round(age), "device": device, "pid": pid, "orphan": orphan, "rerun": new["job_id"]}), flush=True)
        for jid, nid in list(pending.items()):
            r = q / "results" / (nid + ".json")
            if r.exists():
                try:
                    out = json.loads(r.read_text())
                except json.JSONDecodeError:
                    continue
                out.update(job_id=jid, rerun_of_hang=nid)
                tmp = q / "results" / (jid + ".json.tmp")
                tmp.write_text(json.dumps(out))
                tmp.rename(q / "results" / (jid + ".json"))
                del pending[jid]
                print(json.dumps({"resolved": jid, "via": nid, "completed": out.get("completed")}), flush=True)
        if (q / "STOP").exists():
            break
        time.sleep(10)


if __name__ == "__main__":
    main()
