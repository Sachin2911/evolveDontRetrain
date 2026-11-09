"""Frozen online controller with explicit action histories and filter diagnostics."""

from collections import Counter

import torch

from .unisafe09_environment import normalized_to_environment


def backup_arm_action(raw, convention, denominator=1.0):
    """Keep the released six-coordinate postprocessing quirk explicit."""
    result = raw.clone().clamp(-0.5, 0.5)
    if convention == "released":
        # Upstream applies a gripper-style binarization to the last of SIX
        # safety-actor coordinates. In the physical task this is rotation z.
        result[:, -1] = torch.where(result[:, -1] < 0, -1.0, 1.0)
    elif convention != "arm_clip":
        raise ValueError(convention)
    return result / denominator


def model_observation(obs, first=False):
    return {
        **{key: value.clone() for key, value in obs["rgb_camera"].items()},
        # Required by upstream preprocess; excluded from the encoder's input regex.
        "failure": obs["subtask_terms"]["failure"].clone(),
        "success": obs["subtask_terms"]["success"].clone(),
        "eef_pos": obs["policy"]["eef_pos"].clone(),
        "eef_quat": obs["policy"]["eef_quat"].clone(),
        "is_first": torch.full_like(obs["subtask_terms"]["failure"], first),
        "is_last": torch.zeros_like(obs["subtask_terms"]["failure"]),
        "is_terminal": torch.zeros_like(obs["subtask_terms"]["failure"]),
    }


class FrozenController:
    def __init__(self, agent, safety, settings, proposal_adapter=None):
        self.agent, self.safety, self.settings = agent, safety, settings
        self.wm = agent._wm
        self.proposal_adapter = proposal_adapter
        self.reset()

    def reset(self):
        self.task_latent = self.filter_latent = self.previous_executed = None
        self.queries = Counter()
        self.pending_prediction = None
        self.step_index = 0
        if self.proposal_adapter is not None:
            self.proposal_adapter.reset()

    @torch.inference_mode()
    def act(self, obs, mode, diagnostic=False, adapter_observation=None):
        data = self.wm.preprocess(obs)
        embed = self.wm.encoder(data)
        self.queries["encoder"] += 1
        # Original task actor uses sampled posteriors; the filter tracks posterior
        # modes separately. Both histories receive the action actually sent to physics.
        self.task_latent, _ = self.wm.dynamics.obs_step(
            self.task_latent, self.previous_executed, embed, data["is_first"], sample=True
        )
        self.filter_latent, _ = self.wm.dynamics.obs_step(
            self.filter_latent, self.previous_executed, embed, data["is_first"], sample=False
        )
        self.queries["posterior"] += 2
        task_feat = self.wm.dynamics.get_feat(self.task_latent)
        feat = self.wm.dynamics.get_feat(self.filter_latent)
        raw = self.agent._task_behavior.actor(task_feat).mode()
        proposal = raw.clone()
        proposal[:, -1] *= 2
        task_proposal = proposal.clone()
        adapter_log = {}
        if self.proposal_adapter is not None:
            proposal, adapter_log = self.proposal_adapter(
                proposal, obs if adapter_observation is None else adapter_observation, task_feat
            )
        probe = self.settings.get("diagnostic_probe", "none")
        gripper_override = probe == "hold_gripper" and 43 <= self.step_index < 63
        if gripper_override:
            proposal[:, -1] = -1.0
        self.queries["task_actor"] += 1
        uncertainty = self.agent._disag_ensemble.intrinsic_reward_penn(
            torch.cat((feat, proposal), -1)
        )
        imagined = self.wm.dynamics.img_step(self.filter_latent, proposal, sample=True)
        if probe == "modal_next":
            # Consume the original random draw before substituting the mode, so
            # this diagnostic preserves the task policy's RNG schedule.
            imagined["stoch"] = self.wm.dynamics.get_dist(imagined).mode()
        next_feat = self.wm.dynamics.get_feat(imagined)
        next_best, _ = self.safety.select_action(next_feat, eval_mode=True)
        q1, q2 = (
            self.safety.critic1(next_feat, next_best),
            self.safety.critic2(next_feat, next_best),
        )
        backup, _ = self.safety.select_action(feat, eval_mode=True)
        self.queries.update(
            uncertainty_ensemble=1, imagined_step=1, safety_actor=2, safety_critic=2
        )
        uncertainty_trigger = uncertainty.item() > self.settings["uncertainty_threshold"]
        value_trigger = min(q1.item(), q2.item()) < self.settings["reachability_threshold"]
        would_intervene = uncertainty_trigger or value_trigger
        candidate = proposal.clone()
        convention = "arm_clip" if probe == "arm_clip" else self.settings["backup_postprocess"]
        candidate[:, :6] = backup_arm_action(backup, convention, self.settings["action_denom"])
        suppressed = probe == "no_filter" and self.step_index >= 50
        intervened = mode == "filtered" and would_intervene and not suppressed
        selected = candidate if intervened else proposal.clone()
        environment_action = normalized_to_environment(selected)
        executed = environment_action / 0.15
        self.previous_executed = executed.clone()
        failure_probability = self.wm.heads["failure"](feat).mean.item()
        self.queries["failure_head"] += 1
        row = {
            **adapter_log,
            "raw_task_action": raw[0].tolist(),
            "task_proposed_action": task_proposal[0].tolist(),
            "diagnostic_probe": probe,
            "probe_gripper_override": gripper_override,
            "probe_filter_suppressed": suppressed,
            "backup_postprocess": convention,
            "proposed_action": proposal[0].tolist(),
            "backup_raw_arm_action": backup[0].tolist(),
            "backup_candidate_action": candidate[0].tolist(),
            "filtered_action": selected[0].tolist(),
            "executed_normalized_action": executed[0].tolist(),
            "environment_action": environment_action[0].tolist(),
            "uncertainty": uncertainty.item(),
            "next_q1": q1.item(),
            "next_q2": q2.item(),
            "next_min_q": min(q1.item(), q2.item()),
            "uncertainty_trigger": uncertainty_trigger,
            "value_trigger": value_trigger,
            "would_intervene": would_intervene,
            "intervened": intervened,
            "failure_prediction": failure_probability,
            "action_change_l2": (executed - proposal).norm().item(),
            "clipped": bool((selected - executed).abs().max() > 1e-6),
        }
        self.pending_prediction = None
        if diagnostic:
            recon = {k: v.mode()[:, 0] for k, v in self.wm.heads["decoder"](feat[:, None]).items()}
            prediction = self.wm.dynamics.img_step(self.filter_latent, executed, sample=False)
            self.pending_prediction = {
                k: v.mode()[:, 0]
                for k, v in self.wm.heads["decoder"](
                    self.wm.dynamics.get_feat(prediction)[:, None]
                ).items()
            }
            row["reconstruction_image_mse"] = (
                sum((recon[k] - data[k]).square().mean().item() for k in ("front_cam", "wrist_cam"))
                / 2
            )
            self.queries.update(decoder=2, imagined_step=1)
        for value in (proposal, candidate, environment_action, feat, uncertainty, q1, q2):
            if not torch.isfinite(value).all():
                raise RuntimeError("Non-finite online controller output")
        self.step_index += 1
        return environment_action, row

    @torch.inference_mode()
    def prediction_error(self, next_obs):
        if self.pending_prediction is None:
            return {}
        data = self.wm.preprocess(next_obs)
        pred = self.pending_prediction
        return {
            "executed_next_image_mse": sum(
                (pred[k] - data[k]).square().mean().item() for k in ("front_cam", "wrist_cam")
            )
            / 2,
            "executed_next_eef_error_m": (pred["eef_pos"] - data["eef_pos"]).norm().item(),
        }
