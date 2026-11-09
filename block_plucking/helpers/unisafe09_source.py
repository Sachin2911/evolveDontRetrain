"""Build a reproducible compatibility copy; never edit the pinned upstream checkout."""

import ast
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_REVISION = "ab0fdf87f38edefa48b54a291ec5b7156c17b6e0"


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_if_changed(path, text):
    """Keep verified runtime sources stable and never expose partial Python files."""
    if path.exists() and path.read_text() == text:
        return
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(text)
    temporary.replace(path)


def prepare_source():
    upstream = ROOT / "third_party/unisafe09/UNISafe"
    if not upstream.exists():
        upstream.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "https://github.com/CMU-IntentLab/UNISafe.git", str(upstream)],
            check=True,
        )
        subprocess.run(["git", "-C", str(upstream), "checkout", UPSTREAM_REVISION], check=True)
    revision = subprocess.check_output(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != UPSTREAM_REVISION:
        raise RuntimeError(f"Expected UNISafe {UPSTREAM_REVISION}; found {revision}")
    if subprocess.check_output(["git", "-C", str(upstream), "status", "--porcelain"]):
        raise RuntimeError("Upstream checkout must be clean before preparing compatibility copy.")
    runtime = ROOT / "data/unisafe09/runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    manifest = {"revision": revision, "files": []}
    for source in sorted(upstream.rglob("*.py")):
        if ".git" in source.parts:
            continue
        rel = source.relative_to(upstream)
        original = source.read_text()
        text = original
        for old, new in (
            ("omni.isaac.lab_assets", "isaaclab_assets"),
            ("omni.isaac.lab_tasks", "isaaclab_tasks"),
            ("omni.isaac.lab", "isaaclab"),
            ("omni.isaac.core", "isaacsim.core"),
        ):
            text = text.replace(old, new)
        if str(rel).startswith("takeoff/"):
            text = text.replace(
                "isaaclab_tasks.manager_based.manipulation.stack", "isaaclab_tasks.contrib.stack"
            )
            text = text.replace(
                "from isaaclab_assets.franka import FRANKA_PANDA_HIGH_PD_CFG",
                "from isaaclab_assets.robots.franka import FRANKA_PANDA_LEGACY_HIGH_PD_CFG as FRANKA_PANDA_HIGH_PD_CFG",
            ).replace(
                "from isaaclab_assets.franka import FRANKA_PANDA_CFG",
                "from isaaclab_assets.robots.franka import FRANKA_PANDA_LEGACY_CFG as FRANKA_PANDA_CFG",
            )
            text = text.replace(
                "from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg",
                "from isaaclab_physx.sim.schemas import PhysxRigidBodyPropertiesCfg as RigidBodyPropertiesCfg",
            )
            if "sim_utils.RigidBodyPropertiesCfg" in text:
                text = "from isaaclab_physx.sim.schemas import PhysxRigidBodyPropertiesCfg\n" + text
                text = text.replace(
                    "sim_utils.RigidBodyPropertiesCfg", "PhysxRigidBodyPropertiesCfg"
                )
            if "self.sim.physx" in text:
                text = "from isaaclab_physx.physics import PhysxCfg\n" + text
                text = text.replace("self.sim.physx", "self.sim.physics")
                text = text.replace(
                    "\t\tself.sim.physics.bounce_threshold_velocity = 0.2",
                    "\t\tself.sim.physics = PhysxCfg()\n\t\tself.sim.physics.bounce_threshold_velocity = 0.2",
                )
        if str(rel).startswith("takeoff/mdp/"):
            # Isaac Lab 3 exposes backend-neutral array proxies. Select the torch
            # view explicitly for the original task's torch-based MDP functions.
            tensor_fields = (
                "default_joint_pos|default_joint_vel|default_root_state|joint_pos|"
                "soft_joint_pos_limits|object_pos_w|object_quat_w|root_link_pos_w|"
                "root_link_quat_w|root_link_state_w|root_quat_w|target_pos_source|"
                "target_pos_w|target_quat_source"
            )
            text = re.sub(r"(\.data\.(?:" + tensor_fields + r"))\b", r"\1.torch", text)
        if str(rel) == "takeoff/mdp/events.py":
            # Full resets now pass slice(None), while upstream expects explicit IDs.
            text = text.replace(
                "    # poses\n",
                "    if env_ids is None:\n"
                "        env_ids = torch.arange(env.num_envs, device=env.device)\n"
                "    elif isinstance(env_ids, slice):\n"
                "        env_ids = torch.arange(env.num_envs, device=env.device)[env_ids]\n"
                "    # poses\n",
            )
        if str(rel) in (
            "takeoff/takeoff_env_cfg.py",
            "takeoff/config/franka/takeoff_joint_pos_env_cfg.py",
            # Added 1 November 2026 for the SafeBlock probes: the Hard variant needs
            # the same convention fixes. The regular files' output is unchanged.
            "takeoff/hard_takeoff_env_cfg.py",
            "takeoff/config/franka/hard_takeoff_joint_pos_env_cfg.py",
        ):
            # Explicit upstream rotations are wxyz. Isaac Lab 3 configs and
            # internal state use xyzw; defaults already use the new convention.
            def convert_rotation(match):
                w, x, y, z = ast.literal_eval(match[1])
                return "rot=" + repr((x, y, z, w))

            text = re.sub(r"rot=(\([^()]+\)|\[[^\[\]]+\])", convert_rotation, text)
        if str(rel) == "takeoff/mdp/observations.py":
            # Keep exported observations in the checkpoint's original wxyz order.
            text = text.replace(
                "return ee_frame_quat\n", "return ee_frame_quat[..., [3, 0, 1, 2]]\n"
            )
            for field in ("root_link_quat_w", "object_quat_w"):
                text = text.replace(
                    f".data.{field}.torch", f".data.{field}.torch[..., [3, 0, 1, 2]]"
                )
        if str(rel) in ("takeoff/takeoff_env_cfg.py", "takeoff/hard_takeoff_env_cfg.py"):
            # Native PhysX Fabric retained startup articulation transforms after
            # reset. The supported tensor-to-renderer path keeps pixels current
            # without adding physical settling steps or changing robot poses.
            text = text.replace(
                "self.rerender_on_reset = True",
                "self.num_rerenders_on_reset = 1\n\t\tself.sim.use_fabric = False",
            )
        if str(rel) == "takeoff/mdp/__init__.py":
            text += "\nfrom helpers.unisafe09_environment import camera_rgb as image\n"
        if str(rel) == "takeoff/mdp/failure.py":
            text = text.replace("import pytorch3d.transforms as transforms\n", "")
            text = text.replace(
                "    left: RigidObject =",
                "    import pytorch3d.transforms as transforms\n    left: RigidObject =",
            )
        if str(rel) == "takeoff/config/franka/__init__.py":
            # Only the normal block-plucking task is in this pilot. Avoid importing
            # unrelated legacy stack/hard-task configurations as registration side effects.
            text = (
                "import gymnasium as gym\n"
                'gym.register(id="Isaac-Takeoff-Franka-IK-Rel-v0", '
                'entry_point="isaaclab.envs:ManagerBasedRLEnv", '
                'kwargs={"env_cfg_entry_point": '
                '"takeoff.config.franka.takeoff_ik_rel_env_cfg:FrankaTakeoffEnvCfg"}, '
                "disable_env_checker=True)\n"
            )
        if str(rel) == "reachability/env/__init__.py":
            text = text.replace("    load_plugin_envs,\n", "")
        if str(rel) == "dreamerv3_torch/tools.py":
            text = text.replace("import isaacsim.core.utils.torch as torch_utils\n", "")
            for signature in (
                "def reset_episode(vecenv, latent_dynamics, seed=None):",
                "def reset_episode_df(vecenv, latent_dynamics, seed=None):",
            ):
                text = text.replace(
                    signature, signature + "\n\timport isaacsim.core.utils.torch as torch_utils"
                )
        destination = runtime / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        write_if_changed(destination, text)
        manifest["files"].append(
            {
                "path": str(rel),
                "upstream_sha256": sha256(source),
                "runtime_sha256": sha256(destination),
                "modified": text != original,
            }
        )
    write_if_changed(runtime / "compatibility.json", json.dumps(manifest, indent=2) + "\n")
    return runtime


def save_implementation(output):
    """Save the small runnable implementation and pinned manifests with each run."""
    output = Path(output) / "implementation"
    for directory, pattern in (
        ("experiments/helpers", "unisafe09*.py"),
        ("experiments/scripts", "unisafe09*.py"),
        ("configs/unisafe09", "*.txt"),
        ("configs/unisafe09", "*.json"),
        ("configs/unisafe09", "checkpoint.json"),
        ("configs/unisafe09", "requirements-platform.txt"),
        ("configs/unisafe09", "uv-overrides.txt"),
        ("data/unisafe09/runtime", "compatibility.json"),
    ):
        for source in (ROOT / directory).glob(pattern):
            destination = output / source.relative_to(ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
