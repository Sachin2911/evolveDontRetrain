"""Latent planners on the frozen Dreamer world model: goal CEM, anchored CEM, anchored rank.

All planners start from the posterior mode state of the world model and roll
candidate action sequences forward with the deterministic prior. Costs are squared
distances between the imagined terminal feature and a target feature. Targets come
from a library of encoded demonstrations (encode_library.py), following
"Aim Short to Reach Far" (Anchored Planning): aim at what a similar demonstration
reached a few steps later instead of at the far goal.
"""

import time

import numpy as np
import torch


def discretise(actions):
    """Six arm coordinates in the normalised range, one gripper sign."""
    result = actions.clone()
    result[..., :6] = result[..., :6].clamp(-1.0, 1.0)
    result[..., 6] = torch.where(result[..., 6] < 0, -1.0, 1.0)
    return result


class Library:
    """Encoded demonstration steps with nearest-neighbour retrieval on the GPU."""

    def __init__(self, path, device, min_step=2):
        with np.load(path) as z:
            self.feats = torch.tensor(z["feats"], device=device, dtype=torch.float32)
            self.actions = torch.tensor(z["actions"], device=device, dtype=torch.float32)
            self.ep = torch.tensor(z["ep"], device=device, dtype=torch.long)
            self.step = torch.tensor(z["step"], device=device, dtype=torch.long)
            self.ep_start = torch.tensor(z["ep_start"], device=device, dtype=torch.long)
            self.ep_len = torch.tensor(z["ep_len"], device=device, dtype=torch.long)
        self.device = device
        self.min_step = min_step
        self.sq_norm = (self.feats**2).sum(-1)
        self.valid = self.step >= min_step
        self.ep_last = self.ep_start + self.ep_len - 1
        self.queries = 0

    @torch.inference_mode()
    def distances(self, feat):
        # |f|^2 - 2 f.x + |x|^2, computed without materialising a difference tensor.
        return self.sq_norm - 2.0 * (self.feats @ feat) + (feat**2).sum()

    @torch.inference_mode()
    def nearest(self, feat, k):
        self.queries += 1
        d = self.distances(feat)
        d = torch.where(self.valid, d, torch.full_like(d, float("inf")))
        values, index = torch.topk(d, k, largest=False)
        return index, values

    def anchor(self, index, lookahead):
        last = self.ep_last[self.ep[index]]
        target = torch.minimum(index + lookahead, last)
        return self.feats[target], target

    def following_actions(self, index, length):
        """The recorded actions that follow step `index`, padded with the last one."""
        last = self.ep_last[self.ep[index]]
        steps = index[:, None] + 1 + torch.arange(length, device=self.device)[None]
        steps = torch.minimum(steps, last[:, None])
        return self.actions[steps]

    @torch.inference_mode()
    def goal_from_nearest_start(self, first_feat):
        """The final feature of the demonstration whose first frame is nearest."""
        self.queries += 1
        d = self.distances(first_feat)
        d = torch.where(self.step == 0, d, torch.full_like(d, float("inf")))
        start = int(torch.argmin(d))
        episode = int(self.ep[start])
        return self.feats[self.ep_last[episode]], episode, float(d[start])


class Rollout:
    def __init__(self, wm):
        self.dynamics = wm.dynamics
        self.steps = 0

    @torch.inference_mode()
    def run(self, state, actions):
        """state: dict of (1, ...) tensors; actions: (N, T, 7). Returns feats (N, T, F)."""
        n = actions.shape[0]
        current = {k: v.expand(n, *v.shape[1:]).contiguous() for k, v in state.items()}
        feats = []
        for t in range(actions.shape[1]):
            current = self.dynamics.img_step(current, actions[:, t], sample=False)
            feats.append(self.dynamics.get_feat(current))
            self.steps += n
        return torch.stack(feats, 1)


class CEMPlanner:
    def __init__(self, rollout, horizon_blocks, block, samples, elites, iterations, init_std, std_floor, seed, device):
        self.rollout = rollout
        self.h, self.block = horizon_blocks, block
        self.samples, self.elites, self.iterations = samples, elites, iterations
        self.init_std, self.std_floor = init_std, std_floor
        self.generator = torch.Generator(device=device).manual_seed(int(seed))
        self.device = device

    @torch.inference_mode()
    def plan(self, state, target):
        tick = time.perf_counter()
        mean = torch.zeros(self.h, 7, device=self.device)
        std = torch.full((self.h, 7), self.init_std, device=self.device)
        history = []
        for _ in range(self.iterations):
            noise = torch.randn(self.samples, self.h, 7, device=self.device, generator=self.generator)
            candidates = discretise(mean[None] + std[None] * noise)
            sequence = candidates.repeat_interleave(self.block, dim=1)
            feats = self.rollout.run(state, sequence)
            cost = ((feats[:, -1] - target[None]) ** 2).sum(-1)
            _, elite_index = torch.topk(cost, self.elites, largest=False)
            elite = candidates[elite_index]
            mean = elite.mean(0)
            std = elite.std(0).clamp_min(self.std_floor)
            history.append(float(cost[elite_index].mean()))
        best = discretise(mean)
        final = self.rollout.run(state, best[None].repeat_interleave(self.block, dim=1))
        final_cost = float(((final[0, -1] - target) ** 2).sum())
        actions = best[0][None].repeat(self.block, 1)
        return actions, {
            "planner": "cem",
            "elite_cost_by_iteration": history,
            "plan_cost": final_cost,
            "plan_seconds": time.perf_counter() - tick,
            "first_block": best[0].tolist(),
        }


class RankPlanner:
    def __init__(self, rollout, block):
        self.rollout = rollout
        self.block = block

    @torch.inference_mode()
    def plan(self, state, target, candidates, bonus_fn=None):
        """candidates: (K, block, 7) recorded action sequences.

        `bonus_fn(sequences, cost, feats)` returns a (K,) bonus subtracted from the cost
        before the argmin (anchor-selection adapters); `feats` are the imagined features
        (K, block, F) of every candidate. Without it the planner's own argmin is
        used; `base_choice` always records what the bare planner would have chosen.
        """
        tick = time.perf_counter()
        sequences = discretise(candidates)
        feats = self.rollout.run(state, sequences)
        cost = ((feats[:, -1] - target[None]) ** 2).sum(-1)
        base = int(torch.argmin(cost))
        best = base if bonus_fn is None else int(torch.argmin(cost - bonus_fn(sequences, cost, feats)))
        self.last_cost = cost
        return sequences[best], {
            "planner": "rank",
            "plan_cost": float(cost[best]),
            "base_plan_cost": float(cost[base]),
            "candidate_cost_min": float(cost.min()),
            "candidate_cost_median": float(cost.median()),
            "candidate_cost_max": float(cost.max()),
            "chosen_candidate": best,
            "base_choice": base,
            "adapter_changed_choice": best != base,
            "plan_seconds": time.perf_counter() - tick,
            "first_block": sequences[best][0].tolist(),
        }


def build(variant, wm, library, config, seed, device):
    """Return (planner, settings) for `cem_goal`, `ap_cem` or `ap_rank`."""
    rollout = Rollout(wm)
    settings = dict(config["common"], **config[variant])
    if variant in ("cem_goal", "ap_cem"):
        planner = CEMPlanner(
            rollout,
            settings["horizon_blocks"],
            settings["block"],
            settings["samples"],
            settings["elites"],
            settings["iterations"],
            settings["init_std"],
            settings["std_floor"],
            seed,
            device,
        )
    elif variant == "ap_rank":
        planner = RankPlanner(rollout, settings["block"])
    else:
        raise ValueError(variant)
    return planner, settings, rollout
