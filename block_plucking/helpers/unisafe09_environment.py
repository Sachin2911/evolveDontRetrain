"""Small adapters preserving UNISafe observation semantics on Isaac Lab 3."""


def camera_rgb(env, sensor_cfg, data_type="rgb", normalize=False):
    if data_type != "rgb" or normalize:
        raise ValueError("This adapter is for UNISafe's unnormalized RGB observations only.")
    sensor = env.scene.sensors[sensor_cfg.name]
    if hasattr(env, "episode_length_buf") and bool((env.episode_length_buf == 0).any()):
        # On this headless RTX stack, render() alone does not capture a camera.
        # Flush two actual captures after reset, with no physics integration.
        env.sim.forward()
        for _ in range(2):
            env.sim.render()
            sensor.update(0.0, force_recompute=True)
    output = sensor.data.output[data_type]
    # Isaac Lab 3 wraps camera arrays in a backend-neutral tensor container.
    return output.torch[..., :3]


def normalized_to_environment(action):
    """UNISafe NormalizeActions followed by DreamerVecEnvWrapper's clipping.

    Six relative Cartesian pose commands followed by a binary gripper command.
    The environment's IK action term subsequently multiplies the six arm values by 0.5.
    """
    return (action * 0.15).clamp(-0.15, 0.15)


def clone_observation(value):
    """Detach observations from simulator-owned camera and state buffers."""
    if isinstance(value, dict):
        return {key: clone_observation(item) for key, item in value.items()}
    return value.clone()


def physical_arrays(env):
    arrays = {
        k: env.scene[k].data.root_link_state_w.torch.cpu().numpy().copy()
        for k in ("base", "cube_1", "cube_2", "cube_3")
    }
    arrays.update(
        {
            k: getattr(env.scene["robot"].data, k).torch.cpu().numpy().copy()
            for k in ("joint_pos", "joint_vel")
        }
    )
    return arrays


def make_environment(cfg, continuation=False):
    """Preserve terminal images/state before Isaac Lab's same-step automatic reset."""
    from isaaclab.envs import ManagerBasedRLEnv

    class RecordedEnvironment(ManagerBasedRLEnv):
        _inside_step = False
        terminal_snapshot = None
        terminal_terms = None
        terminal_arrays = None
        success_seen = False
        suppressed_resets = 0
        automatic_resets = 0

        def step(self, action):
            self.terminal_snapshot = None
            self.terminal_terms = None
            self.terminal_arrays = None
            self._inside_step = True
            try:
                result = super().step(action)
            finally:
                self._inside_step = False
            if self.terminal_snapshot is not None:
                result[4]["final_obs"] = self.terminal_snapshot
            return result

        def _reset_idx(self, env_ids):
            if self._inside_step:
                self.terminal_snapshot = clone_observation(self.observation_manager.compute())
                terms = {
                    name: bool(self.termination_manager.get_term(name).item())
                    for name in self.termination_manager.active_terms
                }
                self.terminal_terms = terms
                self.terminal_arrays = physical_arrays(self)
                violation = bool(self.terminal_snapshot["subtask_terms"]["failure"].item())
                violation |= terms["ee_position"]
                violation |= terms["cube_2_dropping"] or terms["cube_3_dropping"]
                self.success_seen |= terms["success"] and not violation
                if continuation and self.success_seen and not violation:
                    self.suppressed_resets += 1
                    return
                self.automatic_resets += 1
            return super()._reset_idx(env_ids)

    return RecordedEnvironment(cfg=cfg)
