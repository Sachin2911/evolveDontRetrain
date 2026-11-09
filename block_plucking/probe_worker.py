"""One Block Plucking episode (or several in one process) with the frozen system.

Conditions: native_unfiltered, native_filtered (released task actor, filter in shadow
or live), cem_goal, ap_cem, ap_rank (latent planners on the frozen world model, filter
in shadow). Tasks: regular, hard (terminations restored). Optional Stage 1 adapter on
a planner condition: residual (bounded correction + gripper override) or selection
(anchor re-ranking). Records every control.

Fresh mode: one episode per process (default). In-process mode: `--in-process-seeds`
runs further episodes after the first one in the same simulator process.
"""

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

from probe_common import (  # noqa: E402
    ROOT,
    TASKS,
    atomic_json,
    build_cfg,
    cameras,
    configure_host_renderer,
    geometry,
    keepout_flags,
    project_path,
    save_implementation,
    utc,
    versions,
)

from helpers.unisafe09_source import prepare_source  # noqa: E402

PLANNERS = ("cem_goal", "ap_cem", "ap_rank")
NATIVE = ("native_unfiltered", "native_filtered")


class PlannerController:
    """Tracks the posterior mode latent, replans every block, applies an optional adapter."""

    def __init__(self, agent, safety, settings, variant, planner, planner_settings, library, rollout, adapter=None,
                 gate_threshold=None, gate_force_closed=False):
        self.agent, self.safety, self.settings = agent, safety, settings
        # Gated residual (9 Nov 2026): the residual is applied only when the frozen world
        # model's failure-head probability at the replan control exceeds gate_threshold.
        # None keeps the legacy ungated behaviour bitwise. gate_force_closed installs the
        # adapter but never opens the gate (parity check).
        self.gate_threshold = gate_threshold
        self.gate_force_closed = gate_force_closed
        self.gate_blocks = 0
        self.gate_open_blocks = 0
        self.wm = agent._wm
        self.variant, self.planner, self.ps = variant, planner, planner_settings
        self.library, self.rollout = library, rollout
        self.adapter = adapter
        self.latent = None
        self.previous_executed = None
        self.queue = []
        self.step_index = 0
        self.goal = None
        self.goal_info = {}
        self.plans = 0
        self.queries = {"encoder": 0, "posterior": 0, "plans": 0, "retrievals": 0}

    def frozen_candidate_scores(self, feat, feats, sequences):
        """(K, 3) frozen-system scores of each candidate's imagined block.

        Reachability value (min critic) at the imagined terminal state under the safety
        actor's mean action, the world model's failure probability there, and the mean
        ensemble disagreement along the block. All forwards are deterministic; the RNG
        state is saved and restored so the recorded shadow-filter signals stay identical.
        """
        import torch

        cuda_state, cpu_state = torch.cuda.get_rng_state(), torch.get_rng_state()
        terminal = feats[:, -1]
        best, _ = self.safety.select_action(terminal, eval_mode=True)
        q = torch.minimum(self.safety.critic1(terminal, best), self.safety.critic2(terminal, best)).reshape(-1)
        failure = self.wm.heads["failure"](terminal).mean.reshape(-1)
        k = feats.shape[0]
        pre = torch.cat([feat.reshape(1, 1, -1).expand(k, 1, -1), feats[:, :-1]], 1)
        disagreement = self.agent._disag_ensemble.intrinsic_reward_penn(torch.cat([pre, sequences], -1)).reshape(k, -1).mean(1)
        torch.cuda.set_rng_state(cuda_state)
        torch.set_rng_state(cpu_state)
        return torch.stack([q.to(torch.float32), failure.to(torch.float32), disagreement.to(torch.float32)], 1)

    def act(self, obs, mode, policy_obs=None):
        import torch

        with torch.inference_mode():
            return self._act(obs, mode, policy_obs, torch)

    def _act(self, obs, mode, policy_obs, torch):
        import adapters

        data = self.wm.preprocess(obs)
        embed = self.wm.encoder(data)
        self.queries["encoder"] += 1
        self.latent, _ = self.wm.dynamics.obs_step(
            self.latent, self.previous_executed, embed, data["is_first"], sample=False
        )
        self.queries["posterior"] += 1
        feat = self.wm.dynamics.get_feat(self.latent)
        if self.goal is None:
            goal, episode, distance = self.library.goal_from_nearest_start(feat[0])
            self.goal = goal
            self.goal_info = {"goal_episode": episode, "goal_start_distance": distance}
        plan_log = {}
        if not self.queue:
            retrieval = {}
            index = values = None
            if self.variant == "cem_goal":
                target = self.goal
            else:
                index, values = self.library.nearest(feat[0], self.ps["neighbours"])
                anchor, anchor_index = self.library.anchor(index[:1], self.ps["lookahead"])
                target = anchor[0]
                retrieval = {
                    "retrieved_index": int(index[0]),
                    "retrieved_episode": int(self.library.ep[index[0]]),
                    "retrieved_step": int(self.library.step[index[0]]),
                    "retrieval_distance": float(values[0]),
                    "anchor_index": int(anchor_index[0]),
                }
                self.queries["retrievals"] += 1
            if self.variant == "ap_rank":
                candidates = self.library.following_actions(index, self.ps["block"])
                bonus_fn = None
                if self.adapter is not None and self.adapter.kind == "selection":
                    lib = self.library

                    def bonus_fn(sequences, cost, feats=None):
                        return self.adapter.bonus(adapters.segment_features(lib, index, values, sequences, cost))

                elif self.adapter is not None and self.adapter.kind == "selection2":
                    lib = self.library

                    def bonus_fn(sequences, cost, feats):
                        base = adapters.segment_features(lib, index, values, sequences, cost)
                        extra = self.frozen_candidate_scores(feat, feats, sequences)
                        self.last_selection_features = torch.cat([base, extra], 1)
                        return self.adapter.bonus(self.last_selection_features)

                actions, plan_log = self.planner.plan(self.latent, target, candidates, bonus_fn=bonus_fn)
                if self.adapter is not None and self.adapter.kind == "selection2":
                    f = self.last_selection_features
                    plan_log["adapter_features_chosen"] = f[plan_log["chosen_candidate"]].tolist()
                    plan_log["adapter_features_base_choice"] = f[plan_log["base_choice"]].tolist()
                    plan_log["frozen_scores_spread"] = {
                        name: [float(f[:, 5 + j].min()), float(f[:, 5 + j].median()), float(f[:, 5 + j].max())]
                        for j, name in enumerate(adapters.SELECTION2_FEATURES[5:])
                    }
            else:
                actions, plan_log = self.planner.plan(self.latent, target)
            plan_log.update(retrieval)
            plan_log["target_distance_now"] = float(((feat[0] - target) ** 2).sum())
            plan_log["goal_distance_now"] = float(((feat[0] - self.goal) ** 2).sum())
            plan_log["planner_block"] = actions.tolist()
            if self.adapter is not None and self.adapter.kind == "residual" and self.gate_threshold is not None:
                # Deterministic head on the same feat the per-control failure_prediction uses;
                # no RNG is consumed, so a closed gate leaves the episode bitwise unchanged.
                gate_failure = self.wm.heads["failure"](feat).mean.item()
                gate_open = (not self.gate_force_closed) and gate_failure > self.gate_threshold
                self.gate_blocks += 1
                self.gate_open_blocks += int(gate_open)
                plan_log["gate_failure_probability"] = gate_failure
                plan_log["gate_threshold"] = self.gate_threshold
                plan_log["gate_open"] = bool(gate_open)
                if not gate_open:
                    plan_log["adapter_changed_action"] = False
                    plan_log["adapter_arm_correction_linf"] = 0.0
                    plan_log["adapter_gripper_overrides"] = 0
            else:
                gate_open = True
            if self.adapter is not None and self.adapter.kind == "residual" and gate_open:
                previous = (
                    self.previous_executed[0]
                    if self.previous_executed is not None
                    else torch.zeros(7, device=actions.device)
                )
                x = adapters.adapter_inputs(
                    policy_obs["eef_pos"][0],
                    policy_obs["eef_quat"][0],
                    policy_obs["gripper_pos"][0],
                    actions[0],
                    previous,
                    plan_log["plan_cost"],
                    retrieval.get("retrieval_distance", 0.0),
                    self.step_index,
                )
                actions, adapter_log = self.adapter(x, actions)
                plan_log.update(adapter_log)
                plan_log["adapter_inputs"] = x.tolist()
            self.queue = [actions[i] for i in range(actions.shape[0])]
            self.plans += 1
            self.queries["plans"] += 1
        proposal = self.queue.pop(0)[None]
        # Shadow filter signals, computed exactly as the released controller does.
        uncertainty = self.agent._disag_ensemble.intrinsic_reward_penn(torch.cat((feat, proposal), -1))
        imagined = self.wm.dynamics.img_step(self.latent, proposal, sample=True)
        next_feat = self.wm.dynamics.get_feat(imagined)
        next_best, _ = self.safety.select_action(next_feat, eval_mode=True)
        q1 = self.safety.critic1(next_feat, next_best)
        q2 = self.safety.critic2(next_feat, next_best)
        uncertainty_trigger = uncertainty.item() > self.settings["uncertainty_threshold"]
        value_trigger = min(q1.item(), q2.item()) < self.settings["reachability_threshold"]
        failure_probability = self.wm.heads["failure"](feat).mean.item()
        from helpers.unisafe09_environment import normalized_to_environment

        environment_action = normalized_to_environment(proposal)
        executed = environment_action / 0.15
        self.previous_executed = executed.clone()
        row = {
            **plan_log,
            **self.goal_info,
            "planner_variant": self.variant,
            "new_plan": bool(plan_log),
            "proposed_action": proposal[0].tolist(),
            "executed_normalized_action": executed[0].tolist(),
            "environment_action": environment_action[0].tolist(),
            "uncertainty": uncertainty.item(),
            "next_q1": q1.item(),
            "next_q2": q2.item(),
            "next_min_q": min(q1.item(), q2.item()),
            "uncertainty_trigger": uncertainty_trigger,
            "value_trigger": value_trigger,
            "would_intervene": uncertainty_trigger or value_trigger,
            "intervened": False,
            "failure_prediction": failure_probability,
            "task_proposed_action": proposal[0].tolist(),
            "filtered_action": proposal[0].tolist(),
        }
        for value in (proposal, environment_action, feat, uncertainty, q1, q2):
            if not torch.isfinite(value).all():
                raise RuntimeError("Non-finite planner controller output")
        self.step_index += 1
        return environment_action, row


def build_adapter(args, device):
    import adapters

    if args.adapter == "none":
        return None, None
    if args.condition not in PLANNERS:
        raise SystemExit("Adapters need a planner condition")
    if args.adapter in ("selection", "selection2") and args.condition != "ap_rank":
        raise SystemExit("The selection adapters need ap_rank")
    if args.zero_genotype:
        params = [0.0] * adapters.PARAM_COUNT[args.adapter]
    elif args.genotype_json:
        loaded = json.loads(args.genotype_json.read_text())
        params = [float(v) for v in (loaded["params"] if isinstance(loaded, dict) else loaded)]
    else:
        if args.genotype_seed is None:
            raise SystemExit("Give --genotype-seed or --zero-genotype")
        params = adapters.generic_genotype(args.adapter, args.genotype_seed, args.init_scale)
    if args.adapter == "residual":
        adapter = adapters.ResidualAdapter(params, args.bound, args.gripper_bound, device)
    elif args.adapter == "selection2":
        adapter = adapters.SelectionAdapter2(params, device)
    else:
        adapter = adapters.SelectionAdapter(params, device)
    spec = {
        "kind": args.adapter,
        "zero_genotype": bool(args.zero_genotype),
        "genotype_seed": args.genotype_seed,
        "genotype_json": str(args.genotype_json) if args.genotype_json else None,
        "bound": args.bound if args.adapter == "residual" else None,
        "gripper_bound": args.gripper_bound if args.adapter == "residual" else None,
        "init_scale": args.init_scale,
        "params_hash": adapter.params_hash,
        "parameters": len(params),
        "params": params,
    }
    return adapter, spec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=sorted(TASKS), required=True)
    parser.add_argument("--condition", choices=NATIVE + PLANNERS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--library", type=Path,
                        default=project_path("data/safeblock/library/success_only.npz"))
    parser.add_argument("--config", type=Path, default=ROOT / "SafeBlock/configs/probes-20261101.json")
    parser.add_argument("--freeze", type=Path, help="freeze file to hash into the record")
    parser.add_argument("--no-continuation", action="store_true")
    parser.add_argument("--adapter", choices=("none", "residual", "selection", "selection2"), default="none")
    parser.add_argument("--genotype-seed", type=int)
    parser.add_argument("--zero-genotype", action="store_true")
    parser.add_argument("--genotype-json", type=Path, help='explicit parameters: a JSON list or {"params": [...]}')
    parser.add_argument("--bound", type=float, default=0.5)
    parser.add_argument("--gripper-bound", type=float, default=2.0)
    parser.add_argument("--init-scale", type=float, default=0.5, help="std of the generic initialisation N(0, scale)")
    parser.add_argument("--in-process-seeds", help="comma-separated seeds run after the first episode in the same process")
    parser.add_argument("--gate-threshold", type=float, default=None,
                        help="gated residual: apply the residual only when the failure-head probability at the replan exceeds this (default off = legacy ungated)")
    parser.add_argument("--gate-force-closed", action="store_true", help="install the adapter but never open the gate (parity check)")
    args = parser.parse_args()
    for name in ("output", "library", "config", "freeze", "genotype_json"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, project_path(value))
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    if (output / "worker-start.json").exists():
        raise SystemExit("Every invocation needs a fresh attempt directory")
    continuation = not args.no_continuation
    in_process = [int(s) for s in args.in_process_seeds.split(",")] if args.in_process_seeds else []
    job = {
        "task": args.task,
        "condition": args.condition,
        "seed": args.seed,
        "policy_seed": args.seed,
        "continuation": continuation,
        "adapter": args.adapter,
        "genotype_seed": args.genotype_seed,
        "zero_genotype": args.zero_genotype,
        "genotype_json": str(args.genotype_json) if args.genotype_json else None,
        "bound": args.bound,
        "gripper_bound": args.gripper_bound,
        "in_process_seeds": in_process,
        "gate_threshold": args.gate_threshold,
        "gate_force_closed": bool(args.gate_force_closed),
    }
    atomic_json(output / "worker-start.json", {"utc": utc(), "pid": os.getpid(), "job": job})
    settings = json.loads(project_path("configs/unisafe09/controller.json").read_text())
    config = json.loads(args.config.read_text())
    freeze_hash = hashlib.sha256(args.freeze.read_bytes()).hexdigest() if args.freeze else None
    renderer = configure_host_renderer()
    atomic_json(output / "renderer.json", renderer)
    runtime = prepare_source()
    sys.path.insert(0, str(runtime))
    extra = [args.config, args.library.with_suffix(".json")] + ([args.freeze] if args.freeze else [])
    hashes = save_implementation(output, extra=extra)
    from isaaclab.app import AppLauncher

    started = time.perf_counter()
    launcher = env = None
    account = {"phase": "app_startup", "completed_controls": 0}

    def checkpoint(phase):
        account.update(phase=phase, utc=utc(), elapsed_seconds=time.perf_counter() - started)
        atomic_json(output / "accounting.json", account)

    exit_code = 1
    try:
        checkpoint("app_startup")
        launcher = AppLauncher({"headless": True, "enable_cameras": True})
        import numpy as np
        import torch

        from helpers.unisafe09_controller import FrozenController, model_observation
        from helpers.unisafe09_environment import make_environment, physical_arrays
        from helpers.unisafe09_model import load_frozen
        from helpers.unisafe09_outcomes import EpisodeScore

        import latent_planner

        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        spec = TASKS[args.task]
        cfg = build_cfg(args.task, args.seed)
        env = make_environment(cfg, continuation=continuation)
        checkpoint("model_loading")
        agent, safety, _, provenance = load_frozen(filter_step=settings["filter_step"])
        adapter, adapter_spec = build_adapter(args, "cuda")
        library = latent_planner.Library(args.library, "cuda") if args.condition in PLANNERS else None
        mode = "filtered" if args.condition == "native_filtered" else "unfiltered"

        def make_controller(seed):
            if args.condition in NATIVE:
                controller = FrozenController(agent, safety, settings, None)
                controller.reset()
                return controller
            planner, planner_settings, rollout = latent_planner.build(
                args.condition, agent._wm, library, config["planner"], seed, "cuda"
            )
            return PlannerController(
                agent, safety, settings, args.condition, planner, planner_settings, library, rollout, adapter,
                gate_threshold=args.gate_threshold, gate_force_closed=bool(args.gate_force_closed),
            )

        startup_seconds = None

        def run_episode(seed, out, episode_index):
            nonlocal startup_seconds
            out.mkdir(parents=True, exist_ok=True)
            episode_started = time.perf_counter()
            env.success_seen = False
            env.suppressed_resets = 0
            env.automatic_resets = 0
            torch.cuda.reset_peak_memory_stats()
            physics_before_reset = int(env.sim.get_physics_step_count())
            obs, _ = env.reset(seed=seed)
            assert env.max_episode_length == 300 and abs(env.step_dt - 0.05) < 1e-12
            physics_after_reset = int(env.sim.get_physics_step_count())
            np.savez_compressed(out / "initial_state.npz", **physical_arrays(env))
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            controller = make_controller(seed)
            score = EpisodeScore(continuation)
            rows = []
            (out / "cameras").mkdir(exist_ok=True)
            np.savez_compressed(out / "reset_cameras.npz", cameras=cameras(obs))
            checkpoint("ready")
            if startup_seconds is None:
                startup_seconds = time.perf_counter() - started
            base_xy = None
            keepout_steps = {k: 0 for k in keepout_flags([0, 0, 0], [0, 0])}
            with (out / "steps.jsonl").open("w", buffering=1) as stream:
                for t in range(340 if continuation else 300):
                    before = geometry(obs, spec)
                    if base_xy is None:
                        base_xy = before["cube_positions"][0][:2]
                    image = cameras(obs)
                    np.savez_compressed(out / "cameras" / f"{t:04d}.npz", cameras=image)
                    model_obs = model_observation(obs, first=t == 0)
                    torch.cuda.synchronize()
                    tick = time.perf_counter()
                    if args.condition in NATIVE:
                        action, row = controller.act(
                            model_obs, mode, adapter_observation={"gripper_pos": obs["policy"]["gripper_pos"]}
                        )
                    else:
                        action, row = controller.act(model_obs, mode, policy_obs=obs["policy"])
                    torch.cuda.synchronize()
                    row["inference_seconds"] = time.perf_counter() - tick
                    checkpoint("physics_started")
                    next_obs, reward, terminated, truncated, info = env.step(action)
                    account["completed_controls"] += 1
                    done = bool((terminated | truncated).item())
                    if done and env.terminal_snapshot is None:
                        raise RuntimeError("Terminal observation was not captured")
                    observed = info["final_obs"] if done else next_obs
                    terms = (
                        env.terminal_terms
                        if done
                        else {
                            name: bool(env.termination_manager.get_term(name).item())
                            for name in env.termination_manager.active_terms
                        }
                    )
                    after = geometry(observed, spec)
                    physical_failure = (
                        after["physical_failure"] or terms["cube_2_dropping"] or terms["cube_3_dropping"]
                    )
                    if terms["success"] != after["physical_success"]:
                        raise RuntimeError("Termination and observation success disagree")
                    finished = score.advance(
                        success=terms["success"], physical_failure=physical_failure, boundary=terms["ee_position"]
                    )
                    if done and score.benchmark is None:
                        raise RuntimeError("Premature timeout or an unscored termination")
                    flags = keepout_flags(after["eef_pos"], base_xy)
                    for name, flag in flags.items():
                        keepout_steps[name] += int(flag)
                    row.update(
                        step=t,
                        simulation_time_s=(t + 1) * env.step_dt,
                        reward=reward.item(),
                        terminated=bool(terminated.item()),
                        truncated=bool(truncated.item()),
                        termination_terms=terms,
                        before=before,
                        after=after,
                        keepout=flags,
                        camera_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
                        episode_length=int(env.episode_length_buf.item()),
                        suppressed_resets=env.suppressed_resets,
                        automatic_resets=env.automatic_resets,
                    )
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                    rows.append(row)
                    checkpoint("control_completed")
                    if score.benchmark and not finished and env.automatic_resets:
                        raise RuntimeError("Continuation suffered a premature automatic reset")
                    obs = next_obs
                    if finished:
                        np.savez_compressed(out / "terminal_cameras.npz", cameras=cameras(observed))
                        arrays = env.terminal_arrays if done else physical_arrays(env)
                        np.savez_compressed(out / "terminal_state.npz", **arrays)
                        break
                else:
                    raise RuntimeError("Scoring did not finish within its declared maximum controls")
            measured = int(env.sim.get_physics_step_count())
            plan_rows = [r for r in rows if r.get("new_plan")]
            adapter_rows = [r for r in plan_rows if "adapter_changed_action" in r or "adapter_changed_choice" in r]
            result = {
                "status": "complete",
                "job": {**job, "seed": seed, "policy_seed": seed},
                "episode_index": episode_index,
                "mode": "in_process" if in_process else "fresh_process",
                **score.summary(),
                "interventions": sum(r["intervened"] for r in rows),
                "would_intervene_steps": sum(r["would_intervene"] for r in rows),
                "uncertainty_trigger_steps": sum(r["uncertainty_trigger"] for r in rows),
                "value_trigger_steps": sum(r["value_trigger"] for r in rows),
                "keepout_steps": keepout_steps,
                "keepout_episode": {k: v > 0 for k, v in keepout_steps.items()},
                "keepout_first_step": {
                    k: next((r["step"] for r in rows if r["keepout"][k]), None) for k in keepout_steps
                },
                "failure_geometry_disagreements": sum(not r["after"]["failure_geometry_agrees"] for r in rows),
                "top_displaced_steps": sum(r["after"]["top_displaced"] for r in rows),
                "closing_proposals": sum(r["proposed_action"][-1] < 0 for r in rows),
                "eef_path_length_m": sum(
                    float(np.linalg.norm(np.array(r["after"]["eef_pos"]) - r["before"]["eef_pos"])) for r in rows
                ),
                "eef_max_z": max(r["after"]["eef_pos"][2] for r in rows),
                "max_top_base_xy_distance_m": max(r["after"]["top_base_xy_distance_m"] for r in rows),
                "max_middle_displacement_m": max(
                    float(
                        np.linalg.norm(
                            np.array(r["after"]["cube_positions"][1]) - rows[0]["before"]["cube_positions"][1]
                        )
                    )
                    for r in rows
                ),
                "plans": len(plan_rows),
                "plan_seconds_mean": float(np.mean([r["plan_seconds"] for r in plan_rows])) if plan_rows else None,
                "plan_cost_mean": float(np.mean([r["plan_cost"] for r in plan_rows])) if plan_rows else None,
                "adapter": adapter_spec,
                "adapter_blocks": len(adapter_rows),
                "adapter_blocks_changed": sum(
                    bool(r.get("adapter_changed_action") or r.get("adapter_changed_choice")) for r in adapter_rows
                ),
                "adapter_gripper_overrides": sum(r.get("adapter_gripper_overrides", 0) for r in adapter_rows),
                "adapter_arm_correction_mean_linf": (
                    float(np.mean([r["adapter_arm_correction_linf"] for r in adapter_rows if "adapter_arm_correction_linf" in r]))
                    if any("adapter_arm_correction_linf" in r for r in adapter_rows)
                    else None
                ),
                "adapter_choice_changes": sum(bool(r.get("adapter_changed_choice")) for r in plan_rows),
                "gate_threshold": args.gate_threshold,
                "gate_force_closed": bool(args.gate_force_closed),
                "gate_blocks": getattr(controller, "gate_blocks", 0),
                "gate_open_blocks": getattr(controller, "gate_open_blocks", 0),
                "gate_open_fraction": (
                    getattr(controller, "gate_open_blocks", 0) / getattr(controller, "gate_blocks", 0)
                    if getattr(controller, "gate_blocks", 0) else None
                ),
                "failure_prediction_max": max((r["failure_prediction"] for r in rows if "failure_prediction" in r), default=None),
                "failure_prediction_at_plans": [r["failure_prediction"] for r in plan_rows if "failure_prediction" in r],
                "next_min_q_min": min((r["next_min_q"] for r in rows if "next_min_q" in r), default=None),
                "goal_info": getattr(controller, "goal_info", {}),
                "queries": dict(getattr(controller, "queries", {})),
                "rollout_steps": getattr(getattr(controller, "rollout", None), "steps", 0),
                "physics_steps": measured,
                "physics_before_reset": physics_before_reset,
                "physics_after_reset": physics_after_reset,
                "rollout_physics_steps": measured - physics_after_reset,
                "automatic_resets": env.automatic_resets,
                "suppressed_resets": env.suppressed_resets,
                "startup_seconds": startup_seconds,
                "episode_seconds": time.perf_counter() - episode_started,
                "elapsed_seconds": time.perf_counter() - started,
                "settings": settings,
                "planner_config": config["planner"] if args.condition in PLANNERS else None,
                "freeze_sha256": freeze_hash,
                "model_provenance": provenance,
                "implementation_hashes": hashes,
                "renderer": renderer,
                "platform": versions(),
                "gpu": torch.cuda.get_device_name(),
                "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                "final_state": rows[-1]["after"],
            }
            atomic_json(out / "summary.json", result)
            print(
                json.dumps(
                    {
                        k: result[k]
                        for k in (
                            "outcome", "safe_success", "violation", "steps", "sustained_safe_success",
                            "adapter_blocks_changed", "adapter_gripper_overrides", "plans", "episode_seconds",
                        )
                    }
                    | {"seed": seed}
                ),
                flush=True,
            )
            return result

        if not in_process:
            run_episode(args.seed, output, 1)
        else:
            episodes = []
            first = run_episode(args.seed, output / f"episode-01-warmup-{args.seed}", 1)
            episodes.append({"seed": args.seed, "dir": f"episode-01-warmup-{args.seed}", "warmup": True, "outcome": first["outcome"]})
            for i, seed in enumerate(in_process, start=2):
                name = f"episode-{i:02d}-{seed}"
                r = run_episode(seed, output / name, i)
                episodes.append({"seed": seed, "dir": name, "warmup": False, "outcome": r["outcome"], "safe_success": r["safe_success"], "violation": r["violation"], "steps": r["steps"], "sustained_safe_success": r["sustained_safe_success"]})
            atomic_json(
                output / "summary.json",
                {"status": "complete", "mode": "in_process", "outcome": "multi", "job": job, "episodes": episodes, "elapsed_seconds": time.perf_counter() - started},
            )
        exit_code = 0
    except Exception as error:  # noqa: BLE001
        import traceback

        atomic_json(output / "worker-error.json", {"utc": utc(), "error": repr(error), "traceback": traceback.format_exc()})
        print("WORKER FAILED:", repr(error), flush=True)
        traceback.print_exc()
    finally:
        checkpoint("closing")
        try:
            if env is not None:
                env.close()
        finally:
            if launcher is not None:
                launcher.app.close()
    os._exit(exit_code)


if __name__ == "__main__":
    main()
