"""Graded study (9 Nov 2026): file-queue worker for LeWM + Anchored Planning (tworoom, pusht, reacher).

  python -m graded.ap_worker <tworoom|pusht|reacher> --device cuda:N --queue <root>/<env>

One process per (env, GPU). Jobs and outcomes follow graded/INTERFACES.md. Also serves the old
`{"kind": "cases", "splits": [...]}` job (clean qd_adapters cases, as residual.worker does).
PushT runs with the monitor's opt-in pusher-only switch on (the T never counts).
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np

from qd_adapters import monitor as qd_monitor
from qd_adapters.io import Ledger, read, write
from qd_adapters.runtime import Runner

from graded import ap_plugin

PHASE = "graded"


def run_job(runner, job, queue):
    env_name = runner.task
    case = job["case"]
    kind = job["kind"]
    if kind not in ("native", "steer", "choose", "stack"):
        raise ValueError("unknown kind " + repr(kind))
    body = ap_plugin.Body(env_name)
    ap_plugin.begin(env_name, job, case, body)
    started = time.monotonic()
    try:
        outcome = runner.episode(case["ap_case"], {"id": kind, "kind": "native"}, job["job_id"], PHASE)
    finally:
        ap_plugin.end()
    ctx = ap_plugin.CTX
    step = float(case["step"])
    positions = np.asarray(ctx["positions"], dtype=np.float64)
    goal = ctx["goal"] if ctx["goal"] is not None else ctx.get("true_goal")
    final_goal = float(np.linalg.norm(positions[-1] - goal) / step) if goal is not None and len(positions) else None
    steer, choose = ctx["steer"], ctx["choose"]
    out = {
        "job_id": job["job_id"], "env": env_name, "case_id": case.get("id"), "kind": kind, "completed": True,
        "success": bool(outcome["success"]), "violation": bool(outcome["violation"]),
        "safe_completion": bool(outcome["safe_completion"]),
        "min_clear_steps": float(outcome["swept_min_clearance"]) / step,
        "final_goal_steps": final_goal,
        "steer_open": int(steer.opened) if steer is not None else 0,
        "choose_open": int(choose.opened) if choose is not None else 0,
        "decisions": int(ctx["decisions"]),
        "checks_max": int(max(ctx["checks"], default=0)),
        "parity_sha256": outcome["parity_sha256"], "actions": int(outcome["actions"]),
        "seconds": time.monotonic() - started,
        "swept_min_clearance": float(outcome["swept_min_clearance"]),
        "vetoed": int(ctx["vetoed"]),
        "level": case.get("level"), "split": case.get("split"), "group": case.get("group"),
    }
    if ctx["log_senses"]:
        out["senses_prop"] = ctx["senses_prop"]
        out["senses_warm"] = ctx["senses_warm"]
    if kind in ("steer", "stack"):
        out["body"] = ctx["body_log"]
        out["steer_records"] = ctx["steer_records"]
    if kind in ("choose", "stack"):
        out["choose_records"] = ctx["choose_records"]
    if job.get("record"):
        out["positions"] = positions.tolist()
        media = Path(job.get("media_dir") or (queue / "media"))
        media.mkdir(parents=True, exist_ok=True)
        frames = ctx["frames"]
        np.savez_compressed(
            media / (job["job_id"] + ".npz"),
            frames=np.stack(frames).astype(np.uint8) if frames else np.zeros((0, 1, 1, 3), np.uint8),
            positions=positions,
        )
        out["media"] = str(media / (job["job_id"] + ".npz"))
    return out


def build_cases(runner, splits):
    from residual.worker import build_cases as residual_build_cases

    return residual_build_cases(runner, splits)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("env", choices=("tworoom", "pusht", "reacher"))
    parser.add_argument("--device", required=True)
    parser.add_argument("--queue", required=True)
    parser.add_argument("--idle-exit", type=float, default=6 * 3600.0)
    args = parser.parse_args()
    os.environ["QD_DEVICE"] = args.device
    if args.env == "pusht":
        qd_monitor.PUSHER_ONLY = True  # DESIGN.md: PushT counts the pusher only
    queue = Path(args.queue)
    jobs, results = queue / "jobs", queue / "results"
    jobs.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    tag = args.device.replace(":", "")
    config = read(os.environ["LEWM_CONFIG"])
    runner = Runner(config, args.env)
    runner.ledger = Ledger(queue / f"ledger-{tag}.json", {PHASE: 10**9}, runner.ledger.protocol_sha256)
    ap_plugin.install(runner)
    write(queue / f"ready-{tag}.json", {"device": args.device, "env": args.env, "epoch": time.time(),
                                         "pusher_only": bool(qd_monitor.PUSHER_ONLY)})
    last = time.monotonic()
    try:
        while not (queue / "STOP").exists():
            pending = sorted(jobs.glob("*.json"))
            if not pending:
                if time.monotonic() - last > args.idle_exit:
                    break
                time.sleep(0.5)
                continue
            path = pending[np.random.default_rng().integers(len(pending))]
            claimed = path.with_suffix(f".{tag}.claimed")
            try:
                path.rename(claimed)
            except FileNotFoundError:
                continue
            last = time.monotonic()
            try:
                job = read(claimed)
            except Exception:
                time.sleep(0.2)
                job = read(claimed)
            try:
                if job.get("kind") == "cases":
                    out = {"job_id": job["job_id"], "cases": build_cases(runner, job["splits"])}
                else:
                    out = run_job(runner, job, queue)
                out["worker"] = args.device
            except Exception as error:  # record and continue; never silently rerun
                out = {
                    "job_id": job["job_id"], "env": args.env, "case_id": job.get("case", {}).get("id"),
                    "kind": job.get("kind"), "completed": False, "error": repr(error),
                    "traceback": traceback.format_exc()[-4000:], "worker": args.device,
                }
                print(json.dumps({"job": job["job_id"], "error": repr(error)}), flush=True)
            write(results / (job["job_id"] + ".json"), out)
            if out.get("completed"):
                print(json.dumps(dict(
                    job=out["job_id"], kind=out["kind"], success=out["success"], violation=out["violation"],
                    clear=round(out["min_clear_steps"], 3), actions=out["actions"], steer=out["steer_open"],
                    choose=out["choose_open"], checks=out["checks_max"], s=round(out["seconds"], 1),
                )), flush=True)
    finally:
        runner.close()


if __name__ == "__main__":
    main()
