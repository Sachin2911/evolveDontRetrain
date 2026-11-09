"""One lane of the gated-residual run (9 Nov 2026): claims jobs from a JSONL list and runs
each as a fresh probe_worker process on one GPU.

    CUDA_VISIBLE_DEVICES=0 /venv/main/bin/python SafeBlock/experiments/gated_lane.py \
        --jobs /workspace/safeblock-run/jobs/calibration.jsonl --output /workspace/safeblock-run/calibration --lane 0a

Claims are atomic directory creations, so any number of lanes can share one job list.
A job whose worker leaves no summary.json is retried once (attempt-02). Stops when a
STOP file exists next to the job list.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from probe_common import ROOT, utc  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lane", required=True)
    parser.add_argument("--timeout", type=float, default=1200)
    parser.add_argument("--retries", type=int, default=1)
    args = parser.parse_args()
    jobs = [json.loads(l) for l in args.jobs.read_text().splitlines() if l.strip()]
    args.output.mkdir(parents=True, exist_ok=True)
    ledger = args.output / f"ledger-{args.lane}.jsonl"
    stop = args.jobs.parent / "STOP"
    for job in jobs:
        if stop.exists():
            break
        job_dir = args.output / job["id"]
        try:
            job_dir.mkdir(parents=False, exist_ok=False)
        except FileExistsError:
            continue
        (job_dir / "claimed.json").write_text(json.dumps({"lane": args.lane, "utc": utc(), "gpu": os.environ.get("CUDA_VISIBLE_DEVICES")}))
        for attempt_index in range(1, args.retries + 2):
            attempt = job_dir / f"attempt-{attempt_index:02d}"
            attempt.mkdir()
            command = [
                sys.executable, str(ROOT / "SafeBlock/experiments/probe_worker.py"),
                "--task", job.get("task", "hard"), "--condition", job.get("condition", "ap_rank"),
                "--seed", str(job["seed"]), "--output", str(attempt),
                "--config", str(ROOT / "SafeBlock/configs/probes-20261101.json"),
                "--library", str(ROOT / "SafeBlock/data/safeblock/library/success_only.npz"),
                "--freeze", str(ROOT / "SafeBlock/configs/stage1-freeze.json"),
            ] + list(job.get("args", []))
            started = time.perf_counter()
            entry = {"job": job, "lane": args.lane, "attempt": attempt_index, "started_utc": utc(), "command": command}
            with (attempt / "worker.log").open("w") as log:
                try:
                    done = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT, timeout=args.timeout)
                    entry["exit_code"] = done.returncode
                except subprocess.TimeoutExpired:
                    entry["exit_code"] = "timeout"
            entry["seconds"] = time.perf_counter() - started
            entry["finished_utc"] = utc()
            summary = attempt / "summary.json"
            if summary.exists():
                s = json.loads(summary.read_text())
                entry["result"] = {k: s.get(k) for k in (
                    "outcome", "safe_success", "sustained_safe_success", "violation", "steps", "plans",
                    "adapter_blocks_changed", "gate_blocks", "gate_open_blocks", "startup_seconds", "episode_seconds")}
            with ledger.open("a") as stream:
                stream.write(json.dumps(entry) + "\n")
            print(json.dumps({k: entry.get(k) for k in ("lane", "attempt", "exit_code", "seconds", "result")} | {"id": job["id"]}), flush=True)
            if summary.exists():
                break
    print(f"lane {args.lane} finished", flush=True)


if __name__ == "__main__":
    main()
