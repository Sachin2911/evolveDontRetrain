"""Rented-host AP runner. Only the selected source index changes before native CEM."""

from collections import Counter
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np

from .actors import Scorer
from .features import candidate_features, causal_stream, normalized_clearance, selector_context
from .geometry import SCALES, descriptor, recorded
from .io import Ledger, digest, publish, read, serial
from .library import select
from .monitor import Monitor, PhysicalReader
from . import v4hook


def backend(config, task, require_overlay=True):
    # Reuse reviewed asset and observer helpers without modifying completed studies.
    old = Path(__file__).resolve().parents[1] / "ap_reproduction"
    sys.path.insert(0, str(old))
    from common import configure

    return configure(config, task, require_overlay=require_overlay)


class Runner:
    def __init__(self, config, task):
        self.observer, self.data = None, None
        try:
            self._initialize(config, task)
        except Exception as error:
            counts = dict(self.observer.total) if self.observer is not None else {}
            if self.data is not None:
                self.data.close()
            if self.observer is not None:
                self.observer.close()
            publish(
                "metrics/initialization-failure.json",
                {
                    "task": task,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "counts": counts,
                    "completed": False,
                },
            )
            raise

    def _initialize(self, config, task):
        os.environ["QD_TASK"] = task
        self.config, self.task = config, task
        self.root, runtime = backend(config, task)
        runtime.check_simulator_versions(task)
        from common import assets
        from instrumentation import Observer
        import h5py
        import hdf5plugin  # noqa: F401
        import torch
        from anchored_planning import environments, model
        from anchored_planning.lifecycle import install_cube_renderer_cleanup
        from anchored_planning.native_adapter import NativeAdapter
        from anchored_planning.observation_bank import ObservationBank

        self.torch, self.envs = torch, environments
        self.observer = Observer(config, task)
        self.observer.install()
        self.asset = assets(config, task)
        receipt = read(self.asset["latent_cache"].with_suffix(".receipt.json"))
        if digest(self.asset["latent_cache"]) != receipt["sha256"]:
            raise ValueError("Latent cache checksum differs")
        if digest(self.asset["checkpoint"]) != receipt["checkpoint_sha256"]:
            raise ValueError("Checkpoint checksum differs")
        self.data = h5py.File(self.asset["dataset"], "r")
        if digest(self.asset["dataset"]) != receipt["dataset"]["sha256"]:
            raise ValueError("Dataset checksum differs from the cache source")
        self.states = self.data[self.asset["state_key"]]
        self.latents = np.load(self.asset["latent_cache"], mmap_mode="r", allow_pickle=False)
        self.protocol = read(self.root / "protocols/main.json")
        self.device = torch.device(os.environ.get("QD_DEVICE", "cuda:0"))
        self.model = model.load_model(self.asset["checkpoint"], self.device)
        self.encode = model.encode_retrieval
        self.native = NativeAdapter(
            None,
            {"task": task, "device": self.device, "model": self.model},
            task,
            {"native_action_stats": read(self.root / "protocols/scalers.json")},
            self.root,
        )
        self.bank = ObservationBank(
            {
                "lengths": np.asarray(self.data["ep_len"], np.int64),
                "offsets": np.asarray(self.data["ep_offset"], np.int64),
            },
            self.latents,
            np.asarray(self.protocol["bank_train_episodes"][task], np.int64),
            5,
            np.ones(3, np.float32),
            self.device,
        )
        if task == "cube":
            install_cube_renderer_cleanup(SimpleNamespace(np=np, runtime=environments))
        self.study = Path(config["asset_root"]) / "study" / task
        self.study.mkdir(parents=True, exist_ok=True)
        self.ledger = Ledger(
            self.study / "ledger.json",
            config["action_limits"][task],
            digest(os.environ["LEWM_PROTOCOL"]),
        )
        self.run = Path(os.environ["LEWM_RUN_DIR"])
        self.started = time.monotonic()
        publish(
            "artifacts/runtime.json",
            {
                "task": task,
                "versions": runtime.versions(),
                "ap_revision": config["ap_revision"],
                "cache_receipt": receipt,
                "source_hashes": {p.name: digest(p) for p in Path(__file__).parent.glob("*.py")},
                "protocol_sha256": digest(os.environ["LEWM_PROTOCOL"]),
                "config_sha256": digest(os.environ["LEWM_CONFIG"]),
            },
        )

    def episode(
        self, case, member, identity, phase, *, substeps=True, action_limit=None, selector=None
    ):
        from anchored_planning.planners import TargetGaussian

        cap = case["action_cap"] if action_limit is None else min(action_limit, case["action_cap"])
        self.ledger.reserve(identity, phase, cap)
        self.observer.case, self.observer.phase = Counter(), "setup"
        self.observer.case_limits = {
            "policy_actions_attempted": cap,
            "predictor_calls": ((cap + 4) // 5) * 31,
            "predictor_candidate_blocks": ((cap + 4) // 5) * 9002,
        }
        env, monitor, complete, outcome = None, None, False, None
        relative = "artifacts/" + self.task + "/episodes/" + identity + ".jsonl.gz"
        destination = self.run / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        start = time.monotonic()
        trace_hash = hashlib.sha256()
        actions, decisions, selected_ranks = 0, 0, []

        def log(stream, row, *, parity=False):
            encoded = json.dumps(serial(row), sort_keys=True, allow_nan=False) + "\n"
            stream.write(encoded)
            stream.flush()
            if parity:
                trace_hash.update(encoded.encode())

        try:
            with gzip.open(destination, "xt") as stream:
                row = case["row"]
                state_rows = {
                    i: np.asarray(self.states[i], np.float32)
                    for i in (row["global_start"], row["global_goal"])
                }
                self.observer.guard()
                self.observer.add("resets_attempted")
                env, obs, info, true_goal = self.envs._make_live(
                    {"task": self.task, "arrays": {"states": state_rows}}, row
                )
                self.observer.add("resets_completed")
                reader = PhysicalReader(self.task, env)
                monitor = Monitor(reader, case["hazard"])
                if substeps:
                    monitor.install()
                rgb = np.ascontiguousarray(env.render())
                goal_rgb = np.ascontiguousarray(self.data["pixels"][row["global_goal"]])
                goal_z = self.encode(self.model, goal_rgb[None], self.device)
                current_z = self.encode(self.model, rgb[None], self.device)
                context = selector_context(
                    self.task,
                    current_z,
                    goal_z,
                    reader.shape(),
                    recorded(self.task, true_goal),
                    case["hazard"],
                )
                if selector is not None:
                    chosen = select(selector["members"], selector["predictors"], context)
                    member = selector["policies"][chosen]
                member_id = member["id"]
                actor = (
                    Scorer(member["architecture"], member["parameters"])
                    if member["kind"] == "learned"
                    else None
                )
                gaussian = TargetGaussian(self.native, self.data["pixels"])
                gaussian.set_action_bounds(env.action_space.low, env.action_space.high)
                generator = self.torch.Generator(device=self.device).manual_seed(
                    row["optimizer_seed"]
                )
                previous_ref, previous_rank = None, 0
                v4_rng = np.random.default_rng(row["optimizer_seed"] + 7919)
                v4_records = []
                success = bool(
                    self.envs._success_now(self.task, obs, info, true_goal, False, False)
                )
                initial_success = success
                stopped = False
                log(
                    stream,
                    {
                        "kind": "initial",
                        "case": case,
                        "member": member_id,
                        "context": context,
                        "initial_success": success,
                        "true_goal": true_goal,
                        "initial_geometry": monitor.previous,
                    },
                )
                log(
                    stream,
                    {
                        "kind": "initial_state",
                        "observation": obs,
                        "info": info,
                        "snapshot": self.envs._snapshot(self.task, env),
                        "rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
                    },
                    parity=True,
                )
                with self.torch.inference_mode():
                    while not success and not stopped and actions < cap:
                        self.observer.guard()
                        if actions:
                            rgb = np.ascontiguousarray(env.render())
                            current_z = self.encode(self.model, rgb[None], self.device)
                        shape = reader.shape()
                        fraction = (case["action_cap"] - actions) / case["action_cap"]
                        rank = 0
                        self.observer.phase = "policy"
                        before_calls = self.observer.case["predictor_calls"]
                        before_blocks = self.observer.case["predictor_candidate_blocks"]
                        if member["kind"] == "replay":
                            proposed = np.asarray(
                                self.data["action"][
                                    row["global_start"] + actions : row["global_start"]
                                    + actions
                                    + 5
                                ]
                            )
                            proposed = np.clip(
                                proposed, env.action_space.low, env.action_space.high
                            )
                            detail = {"replay": True}
                            support = []
                        else:
                            support, costs = self.bank.nearest_topk(
                                current_z, goal_z, max(5, row["horizon"] - actions), 8
                            )
                            support, costs = support[0], costs[0]
                            if member["kind"] == "v4":
                                rank = 0
                                v4hook.begin_decision(
                                    task=self.task,
                                    rule=member["rule"],
                                    hazard=case["hazard"],
                                    env=env,
                                    step=actions,
                                    rng=v4_rng,
                                    genotype=member.get("genotype"),
                                )
                            elif member["kind"] == "rank":
                                rank = member["rank"]
                            elif member["kind"] == "heuristic" or (
                                actor is not None and not actor.neutral
                            ):
                                features = candidate_features(
                                    self.task,
                                    support,
                                    costs,
                                    current_z,
                                    goal_z,
                                    self.latents,
                                    self.states,
                                    shape,
                                    case["hazard"],
                                    fraction,
                                )
                                if actor is None:
                                    rank = int(np.argmax(features[:, 11]))
                                else:
                                    actor.observe(
                                        causal_stream(
                                            self.task,
                                            shape["reference"],
                                            previous_ref,
                                            previous_rank,
                                            normalized_clearance(
                                                shape, case["hazard"], SCALES[self.task]
                                            ),
                                            fraction,
                                        ),
                                        decisions,
                                    )
                                    rank = actor.choose(features)
                            # Native and zero-head policies reach this call with the exact
                            # original support array and no extra model or RNG calls.
                            selected = (
                                support
                                if rank == 0
                                else np.r_[support[rank], np.delete(support, rank)]
                            )
                            proposed, detail = gaussian.choose(
                                "gaussian_observed", rgb, selected, generator, row["global_goal"]
                            )
                            if (
                                self.observer.case["predictor_calls"] - before_calls,
                                self.observer.case["predictor_candidate_blocks"] - before_blocks,
                            ) != (31, 9002):
                                raise AssertionError("Native CEM work changed")
                        previous_ref, previous_rank = shape["reference"].copy(), rank
                        selected_ranks.append(rank)
                        log(
                            stream,
                            {
                                "kind": "decision",
                                "step": actions,
                                "rank": rank,
                                "sources": support,
                                "detail": detail,
                            },
                        )
                        log(
                            stream,
                            {
                                "kind": "proposal",
                                "step": actions,
                                "actions": proposed,
                                "rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
                                "generator_sha256": hashlib.sha256(
                                    generator.get_state().cpu().numpy().tobytes()
                                ).hexdigest(),
                            },
                            parity=True,
                        )
                        decisions += 1
                        v4_record = v4hook.take_record() if member["kind"] == "v4" else None
                        block_swept = np.inf
                        for action in proposed:
                            self.observer.add("policy_actions_attempted")
                            obs, reward, terminated, truncated, info = env.step(action.copy())
                            self.observer.add("policy_actions_completed")
                            monitor.observe(controller_boundary=True)
                            block_swept = min(block_swept, monitor.last_swept)
                            actions += 1
                            success = bool(
                                self.envs._success_now(
                                    self.task, obs, info, true_goal, False, False
                                )
                            )
                            stopped = bool((terminated or truncated) and not success)
                            log(
                                stream,
                                {
                                    "kind": "transition",
                                    "action": action,
                                    "observation": obs,
                                    "info": info,
                                    "snapshot": self.envs._snapshot(self.task, env),
                                    "reward": reward,
                                    "terminated": terminated,
                                    "truncated": truncated,
                                    "success": success,
                                },
                                parity=True,
                            )
                            log(
                                stream,
                                {
                                    "kind": "safety",
                                    "step": actions,
                                    "geometry_samples": monitor.take_samples(),
                                    "sampled_min": monitor.sampled_min,
                                    "swept_min": monitor.swept_min,
                                },
                            )
                            if success or stopped or actions >= cap:
                                break
                        if v4_record is not None:
                            v4_record["realised_block_swept"] = float(block_swept)
                            v4_record["success_after_block"] = bool(success)
                            v4_records.append(v4_record)
                outcome = {
                    "identity": identity,
                    "case_id": case["id"],
                    "member": member_id,
                    "source_group": case["source_group"],
                    "condition": case["condition"],
                    "success": success,
                    "initial_success": initial_success,
                    "violation": monitor.violation,
                    "safe_completion": success and not monitor.violation,
                    "unfinished": not success,
                    "actions": actions,
                    "decisions": decisions,
                    "selected_ranks": selected_ranks,
                    "context": context,
                    "sampled_min_clearance": monitor.sampled_min,
                    "swept_min_clearance": monitor.swept_min,
                    "monitor_samples": monitor.samples,
                    "descriptor": descriptor(monitor.points, case["hazard"], self.task),
                    "parity_sha256": trace_hash.hexdigest(),
                    "completed": True,
                    "elapsed_seconds": time.monotonic() - start,
                }
                log(stream, {"kind": "outcome", **outcome})
                outcome["v4_records"] = v4_records
                complete = True
        finally:
            if monitor is not None:
                monitor.close()
            if env is not None:
                env.close()
            counts = dict(self.observer.case)
            self.ledger.finish(identity, counts, complete)
            self.observer.case = None
            from run_io import mark_ready, report_progress

            # The file is already under artifacts; readiness is published only once closed.
            if destination.exists():
                mark_ready(self.run, relative)
            publish(
                "metrics/attempts/" + identity + ".json",
                {
                    "identity": identity,
                    "completed": complete,
                    "counts": counts,
                    "elapsed_seconds": time.monotonic() - start,
                    "outcome": None if outcome is None else {k: v for k, v in outcome.items() if k != "v4_records"},
                },
            )
            report_progress(
                self.ledger.limits[phase] - self.ledger.remaining(phase),
                self.ledger.limits[phase],
                unit=f"{self.task} {phase} charged actions",
            )
            print(
                json.dumps(
                    {
                        "task": self.task,
                        "identity": identity,
                        "completed": complete,
                        "actions": counts.get("policy_actions_attempted", 0),
                        "elapsed_seconds": time.monotonic() - start,
                    }
                ),
                flush=True,
            )
        return outcome

    def close(self):
        self.data.close()
        self.observer.close()
        publish(
            "metrics/accounting.json",
            {
                "task": self.task,
                "counts": self.observer.total,
                "elapsed_seconds": time.monotonic() - self.started,
            },
        )
        ledger = (
            read(self.ledger.path)
            if self.ledger.path.exists()
            else {
                "protocol_sha256": self.ledger.protocol_sha256,
                "limits": self.ledger.limits,
                "attempts": {},
            }
        )
        publish("metrics/ledger.json", ledger)
