"""Compact adapters around the frozen anchored planner (Stage 1 interfaces).

Both adapters are pure functions of their inputs with no learned state. Zero weights
reproduce the bare planner exactly: tanh(0) = 0 leaves the residual at zero and the
gripper sign unchanged; a zero selection weight leaves the planner's own argmin.
"""

import hashlib

import torch

INPUT_COUNT = 26
HIDDEN = 8
RESIDUAL_PARAMS = INPUT_COUNT * HIDDEN + HIDDEN + HIDDEN * 7 + 7  # 279
SELECTION_FEATURES = ("retrieval_distance", "imagined_cost", "mean_abs_arm_action", "closing_fraction", "progress_in_demo")
SELECTION_PARAMS = len(SELECTION_FEATURES)
SELECTION2_FEATURES = SELECTION_FEATURES + ("imagined_min_q", "imagined_failure_probability", "imagined_disagreement")
SELECTION2_PARAMS = len(SELECTION2_FEATURES)
PARAM_COUNT = {"residual": RESIDUAL_PARAMS, "selection": SELECTION_PARAMS, "selection2": SELECTION2_PARAMS}


def generic_genotype(kind, genotype_seed, scale=0.5):
    n = PARAM_COUNT[kind]
    generator = torch.Generator(device="cpu").manual_seed(int(genotype_seed))
    return (torch.randn(n, generator=generator) * scale).tolist()


def genotype_hash(values):
    return hashlib.sha256(",".join(f"{v:.8f}" for v in values).encode()).hexdigest()


def adapter_inputs(eef_pos, eef_quat, gripper_pos, planner_command, previous_executed, plan_cost, retrieval_distance, step):
    """The 26 shared inputs with the fixed scalings declared in the freeze."""
    parts = [
        eef_pos.reshape(-1),
        eef_quat.reshape(-1),
        gripper_pos.reshape(-1) * 25.0,
        planner_command.reshape(-1),
        previous_executed.reshape(-1),
        torch.tensor([plan_cost / 100.0, retrieval_distance / 100.0, step / 300.0], device=eef_pos.device),
    ]
    x = torch.cat([p.to(torch.float32) for p in parts])
    if x.shape[0] != INPUT_COUNT:
        raise ValueError(f"Adapter expects {INPUT_COUNT} inputs, got {x.shape[0]}")
    return x


class ResidualAdapter:
    """Bounded correction on the six arm coordinates and a bounded gripper override."""

    kind = "residual"

    def __init__(self, params, bound, gripper_bound, device):
        values = torch.tensor(params, dtype=torch.float32, device=device)
        if values.numel() != RESIDUAL_PARAMS:
            raise ValueError(f"Residual adapter needs {RESIDUAL_PARAMS} parameters, got {values.numel()}")
        i = 0
        self.w1 = values[i : i + INPUT_COUNT * HIDDEN].reshape(INPUT_COUNT, HIDDEN); i += INPUT_COUNT * HIDDEN
        self.b1 = values[i : i + HIDDEN]; i += HIDDEN
        self.w2 = values[i : i + HIDDEN * 7].reshape(HIDDEN, 7); i += HIDDEN * 7
        self.b2 = values[i : i + 7]
        self.bound = float(bound)
        self.gripper_bound = float(gripper_bound)
        self.params_hash = genotype_hash(params)

    @torch.inference_mode()
    def __call__(self, x, block_actions):
        """x: (26,) inputs; block_actions: (5, 7) planner actions. Returns (5, 7) and a log."""
        h = torch.tanh(x @ self.w1 + self.b1)
        u = torch.tanh(h @ self.w2 + self.b2)
        arm = (block_actions[:, :6] + self.bound * u[:6]).clamp(-1.0, 1.0)
        gripper = torch.where(block_actions[:, 6] + self.gripper_bound * u[6] < 0, -1.0, 1.0)
        out = torch.cat([arm, gripper[:, None]], 1)
        log = {
            "adapter_u": u.tolist(),
            "adapter_arm_correction_linf": float((out[:, :6] - block_actions[:, :6]).abs().max()),
            "adapter_gripper_overrides": int((gripper != torch.where(block_actions[:, 6] < 0, -1.0, 1.0)).sum()),
            "adapter_changed_action": bool((out != block_actions).any()),
        }
        return out, log


class SelectionAdapter:
    """Linear bonus over standardised per-segment features, added to the planner's cost."""

    kind = "selection"
    n_params = SELECTION_PARAMS

    def __init__(self, params, device):
        values = torch.tensor(params, dtype=torch.float32, device=device)
        if values.numel() != self.n_params:
            raise ValueError(f"{self.kind} adapter needs {self.n_params} parameters, got {values.numel()}")
        self.w = values
        self.params_hash = genotype_hash(params)

    @torch.inference_mode()
    def bonus(self, features):
        """features: (K, 5) raw per-segment features. Returns (K,) bonus to subtract from cost."""
        z = (features - features.mean(0, keepdim=True)) / (features.std(0, keepdim=True) + 1e-6)
        return z @ self.w


class SelectionAdapter2(SelectionAdapter):
    """The 5 planner statistics plus three frozen-system scores of each candidate's imagined
    block: reachability value (min of the two critics) at the imagined terminal state, the
    world model's failure probability there, and the uncertainty ensemble's disagreement
    along the block. Zero weights reproduce the bare planner."""

    kind = "selection2"
    n_params = SELECTION2_PARAMS


def segment_features(library, index, values, candidates, cost):
    """(K, 5): retrieval distance, imagined cost, mean |arm|, closing fraction, progress."""
    ep = library.ep[index]
    progress = library.step[index].to(torch.float32) / library.ep_len[ep].to(torch.float32)
    return torch.stack(
        [
            values.to(torch.float32),
            cost.to(torch.float32),
            candidates[:, :, :6].abs().mean((1, 2)),
            (candidates[:, :, 6] < 0).to(torch.float32).mean(1),
            progress,
        ],
        1,
    )
