"""residualRecipe POC worker: one process per GPU, consuming episode jobs from a directory.

Jobs are JSON files {job_id, phase, case, member, native_scale}. member.kind is
'native', 'residual' (genes required), 'parity' (hook installed, gate forced closed).
The planner runs unchanged; the hook wraps TargetGaussian.choose and edits only the
executed block, after the planner has finished, with no extra model or RNG calls.
"""

import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np

from qd_adapters.io import Ledger, read, serial, write
from qd_adapters.panels import contexts, split_groups
from qd_adapters.runtime import Runner

from residual.hook import GatedResidual, make_oracle

CTX = {"active": False, "env": None, "records": [], "decisions": 0, "abs_sum": 0.0}


def install(runner):
    envs = runner.envs
    original_make = envs._make_live

    def make_live(*args, **kwargs):
        out = original_make(*args, **kwargs)
        env = out[0]
        CTX["env"] = env
        if CTX.get("record"):
            oracle = CTX["oracle"]
            original_step = env.step
            try:
                CTX["frames"].append(np.ascontiguousarray(env.render()))
            except Exception:
                pass
            CTX["positions"].append(np.asarray(oracle.position(env), dtype=np.float64).tolist())

            def step(action):
                result = original_step(action)
                try:
                    CTX["frames"].append(np.ascontiguousarray(env.render()))
                except Exception:
                    pass
                CTX["positions"].append(
                    np.asarray(oracle.position(env), dtype=np.float64).tolist()
                )
                return result

            env.step = step
        return out

    envs._make_live = make_live
    from anchored_planning.planners import TargetGaussian

    original_choose = TargetGaussian.choose

    def choose(self, *args, **kwargs):
        proposed, detail = original_choose(self, *args, **kwargs)
        block = np.asarray(proposed)
        CTX["decisions"] += 1
        CTX["abs_sum"] += float(np.abs(block).mean())
        if not CTX["active"]:
            return proposed, detail
        env = CTX["env"]
        cap = CTX["cap"]
        fraction = max(0.0, 1.0 - (CTX["decisions"] - 1) * block.shape[0] / cap)
        out, record = CTX["adapter"].apply(
            env, CTX["hazard"], block, env.action_space.low, env.action_space.high, fraction
        )
        CTX["records"].append(record)
        if isinstance(detail, dict):
            detail = dict(detail, residual=record)
        return out, detail

    TargetGaussian.choose = choose


def build_cases(runner, splits):
    groups = split_groups(
        runner.task,
        runner.data["ep_len"][:],
        runner.data["ep_offset"][:],
        runner.protocol,
        runner.config["prior_reproduction_episodes"].get(runner.task, []),
    )
    cases = []
    for split in splits:
        cases.extend(contexts(runner.task, groups[split], runner.states, split, infeasible=[]))
    return cases


def run_job(runner, job):
    member = job["member"]
    kind = member["kind"]
    CTX.update(active=False, env=None, records=[], decisions=0, abs_sum=0.0)
    CTX.update(record=bool(job.get("record")), frames=[], positions=[], oracle=None)
    if CTX["record"]:
        CTX["oracle"] = make_oracle(runner.task)
    if kind in ("residual", "parity"):
        CTX["adapter"] = GatedResidual(
            member["genes"], runner.task, job["native_scale"], force_closed=(kind == "parity")
        )
        CTX["hazard"] = job["case"]["hazard"]
        CTX["cap"] = job["case"]["action_cap"]
        CTX["active"] = True
    started = time.monotonic()
    outcome = runner.episode(
        job["case"], {"id": member["id"], "kind": "native"}, job["job_id"], "residual"
    )
    outcome.pop("v4_records", None)
    records = CTX["records"]
    outcome.update(
        job_id=job["job_id"],
        phase=job["phase"],
        member_id=member["id"],
        member_kind=kind,
        decisions_seen=CTX["decisions"],
        mean_abs_action=(CTX["abs_sum"] / CTX["decisions"]) if CTX["decisions"] else None,
        gate_open=int(sum(r["gate"] for r in records)),
        vetoed=int(sum(r["veto"] for r in records)),
        hook_decisions=len(records),
        min_m=min((r["m"] for r in records if r["m"] is not None), default=None),
        seconds=time.monotonic() - started,
    )
    if CTX["record"]:
        media = Path(job["media_dir"])
        media.mkdir(parents=True, exist_ok=True)
        frames = CTX["frames"]
        np.savez_compressed(
            media / (job["job_id"] + ".npz"),
            frames=np.stack(frames).astype(np.uint8) if frames else np.zeros((0, 1, 1, 3), np.uint8),
            positions=np.asarray(CTX["positions"], dtype=np.float64),
            gate=np.asarray([r["gate"] for r in records], np.int8),
            veto=np.asarray([r["veto"] for r in records], np.int8),
            m=np.asarray([(-99.0 if r["m"] is None else r["m"]) for r in records], np.float64),
            hazard_center=np.asarray(job["case"]["hazard"]["center"] if job["case"]["hazard"] else [np.nan, np.nan]),
            hazard_radius=np.asarray(job["case"]["hazard"]["radius"] if job["case"]["hazard"] else np.nan),
        )
        outcome["media"] = str(media / (job["job_id"] + ".npz"))
        outcome["records"] = records
    return outcome


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task")
    parser.add_argument("--device", required=True)
    parser.add_argument("--queue", required=True)
    parser.add_argument("--idle-exit", type=float, default=3600.0)
    args = parser.parse_args()
    os.environ["QD_DEVICE"] = args.device
    queue = Path(args.queue)
    jobs, results = queue / "jobs", queue / "results"
    jobs.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    tag = args.device.replace(":", "")
    config = read(os.environ["LEWM_CONFIG"])
    runner = Runner(config, args.task)
    runner.ledger = Ledger(
        queue / f"ledger-{tag}.json", {"residual": 10**9}, runner.ledger.protocol_sha256
    )
    install(runner)
    write(queue / f"ready-{tag}.json", {"device": args.device, "task": args.task, "epoch": time.time()})
    last = time.monotonic()
    try:
        while not (queue / "STOP").exists():
            pending = sorted(p for p in jobs.glob("*.json"))
            if not pending:
                if time.monotonic() - last > args.idle_exit:
                    break
                time.sleep(1.0)
                continue
            # Fair sharing between concurrently running drivers: pick a random pending job.
            path = pending[np.random.default_rng().integers(len(pending))]
            claimed = path.with_suffix(f".{tag}.claimed")
            try:
                path.rename(claimed)
            except FileNotFoundError:
                continue
            last = time.monotonic()
            job = read(claimed)
            try:
                if job.get("kind") == "cases":
                    cases = build_cases(runner, job["splits"])
                    out = {"job_id": job["job_id"], "cases": cases}
                else:
                    out = run_job(runner, job)
                out["worker"] = args.device
            except Exception as error:  # record and continue; never silently rerun
                out = {
                    "job_id": job["job_id"],
                    "phase": job.get("phase"),
                    "member_id": job.get("member", {}).get("id"),
                    "member_kind": job.get("member", {}).get("kind"),
                    "case_id": job.get("case", {}).get("id"),
                    "error": repr(error),
                    "traceback": traceback.format_exc()[-4000:],
                    "completed": False,
                    "worker": args.device,
                }
                print(json.dumps({"job": job["job_id"], "error": repr(error)}), flush=True)
            write(results / (job["job_id"] + ".json"), out)
            if "error" not in out and "cases" not in out:
                print(
                    json.dumps(
                        dict(
                            job=out["job_id"], kind=out["member_kind"], success=out["success"],
                            violation=out["violation"], safe=out["safe_completion"],
                            actions=out["actions"], gate=out["gate_open"], veto=out["vetoed"],
                            s=round(out["seconds"], 1),
                        )
                    ),
                    flush=True,
                )
    finally:
        runner.close()


if __name__ == "__main__":
    main()
