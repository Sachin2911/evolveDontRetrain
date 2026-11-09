"""Construct the released frozen modules without starting Isaac Sim or training."""

import argparse
import json
import sys

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from .unisafe09_source import ROOT, prepare_source, sha256


class Config(argparse.Namespace):
    def __contains__(self, key):
        return hasattr(self, key)


def load_frozen(checkpoints=None, device="cuda", filter_step=200000):
    runtime = prepare_source()
    sys.path[:0] = [str(runtime), str(runtime / "dreamerv3_torch")]
    from dreamerv3_torch import models, uncertainty
    from reachability.policy.SAC_Reachability_Env import SAC

    checkpoints = checkpoints or ROOT / "data/unisafe09/pretrained"
    config_data = json.loads((checkpoints / "filter_config.json").read_text())
    config_data.update(device=device, compile=False)
    # The pickled training config contains an extra escaped backslash. Match camera
    # keys explicitly; strict checkpoint loading below verifies the resulting architecture.
    for part in ("encoder", "decoder"):
        config_data[part]["cnn_keys"] = ".*_cam$"
    config = Config(**config_data)
    spaces = {
        "front_cam": gym.spaces.Box(0, 255, (128, 128, 3), dtype=np.uint8),
        "wrist_cam": gym.spaces.Box(0, 255, (128, 128, 3), dtype=np.uint8),
        "eef_pos": gym.spaces.Box(-np.inf, np.inf, (3,), dtype=np.float32),
        "eef_quat": gym.spaces.Box(-np.inf, np.inf, (4,), dtype=np.float32),
        **{k: gym.spaces.Box(0, 1, (), dtype=bool) for k in ("is_first", "is_last", "is_terminal")},
    }
    actions = gym.spaces.Box(-0.5, 0.5, (7,), dtype=np.float32)
    agent = nn.Module()
    agent._wm = models.WorldModel(gym.spaces.Dict(spaces), actions, 0, config)
    agent._task_behavior = models.ImagBehavior(config, agent._wm)
    agent._expl_behavior = agent._task_behavior
    agent._disag_ensemble = uncertainty.OneStepPredictor(config, agent._wm)
    checkpoint = torch.load(checkpoints / "latest.pt", map_location="cpu", weights_only=True)
    state = {k.replace("._orig_mod", ""): v for k, v in checkpoint["agent_state_dict"].items()}
    if len(state) != len(checkpoint["agent_state_dict"]):
        raise RuntimeError("Compiled-key normalization produced duplicate names.")
    agent.load_state_dict(state, strict=True)
    del checkpoint, state
    agent.to(device).eval().requires_grad_(False)
    safety = SAC(
        config,
        config.dyn_stoch * config.dyn_discrete + config.dyn_deter,
        config.num_actions - 1,
        config.control_net,
        config.critic_net,
    )
    loaded = {}
    for name in ("actor", "critic1", "critic2"):
        weights = torch.load(
            checkpoints / f"filter/model/{name}-{filter_step}.pth",
            map_location=device,
            weights_only=True,
        )
        getattr(safety, name).load_state_dict(weights, strict=True)
        loaded[name] = weights
    # Upstream shares the critics' preprocessing network. Verify both snapshots
    # agree there, so loading critic2 cannot silently replace different critic1 weights.
    for key, value in loaded["critic1"].items():
        if key.startswith("preprocess.") and not torch.equal(value, loaded["critic2"][key]):
            raise RuntimeError(f"Shared critic checkpoint mismatch: {key}")
    del loaded, weights
    safety.eval().requires_grad_(False)
    provenance = {
        "source_revision": json.loads((runtime / "compatibility.json").read_text())["revision"],
        "compatibility_sha256": sha256(runtime / "compatibility.json"),
        "checkpoint_sha256": sha256(checkpoints / "latest.pt"),
        "filter_step": filter_step,
        "filter_sha256": {
            name: sha256(checkpoints / f"filter/model/{name}-{filter_step}.pth")
            for name in ("actor", "critic1", "critic2")
        },
        "filter_config_sha256": sha256(checkpoints / "filter_config.json"),
        "mirror": json.loads((ROOT / "configs/unisafe09/checkpoint.json").read_text()),
        "config_changes": {"compile": False, "device": device, "camera_regex": ".*_cam$"},
        "checkpoint_loading": "strict; only ._orig_mod key components removed",
    }
    return agent, safety, config, provenance
