"""System oracles for the non-planar tasks (Reacher, Cube) used by residual.hook.

Interface (same as hook.PlanarOracle):
  rollout(env, hazard, blocks)   blocks (N, 5, action_dim) raw actions ->
                                 (clear[N], safe_steps[N], path[N, K, 3])
                                 clearance in the task's physical units: the minimum over the
                                 block of the swept distance between the protected body and the
                                 hazard disk minus both radii, defined exactly as
                                 qd_adapters.geometry.swept_clearance does for that task.
  position(env)                  protected point (physical coordinates)
  directions(env, hazard, block) (e, e_perp, d_e, d_perp): unit ACTION-space directions that move
                                 the protected point away from the hazard and perpendicular to
                                 that, and the cosines of the native block's net action with them.

Reacher: the oracle is the simulator itself on a scratch copy of the live dm_control physics
(share_model=True), stepped exactly as the live wrapper steps it (action_scale transform,
`action_repeat` repeats of `n_sub_steps` physics substeps), reading joint angles after every
substep. The fingertip is the analytic two-link point used by geometry.recorded('reacher'),
and the per-substep swept term includes the same chord-error bound as the monitor. Direction
basis: J^T e (Jacobian of the fingertip w.r.t. the two joint angles), normalised.

Cube: OGBench manipspace actions are relative end-effector targets: action[:3] in [-1, 1] is
unnormalised to +-0.05 m per control step (manipspace_env.py, `action_range`), added to the
current pinch-site position and clipped to the workspace bounds; a PD-tracked IK controller then
follows that target for 25 physics substeps. The kinematic oracle is eef += a[:3] * 0.05 *
CUBE_TRACK (then workspace clip), plus the carried-or-static cube box term (geometry.recorded('cube')
also protects the 4 cm cube); CUBE_TRACK is the realised-over-commanded displacement ratio,
to be fitted by regressing realised displacement on action over native episodes (see
validate_oracles.py). Until fitted it is 1.0 (the commanded target), which is conservative for
clearance only if the arm never overshoots. Orientation and gripper actions do not move the
pinch site in this model.
"""

import types

import numpy as np

from qd_adapters.geometry import clearance as shape_clearance, rotation, segment_distance


def _raw_mujoco():
    """The unpatched MuJoCo step functions.

    qd_adapters.monitor.Monitor.install patches mujoco.mj_step/mj_step2 to sample the LIVE
    geometry after every substep. A scratch rollout must not trigger those samples, so the
    originals are recovered from the patch closures when a patch is installed.
    """
    import mujoco

    out = {}
    for name in ("mj_step", "mj_step1", "mj_step2"):
        fn = getattr(mujoco, name)
        while isinstance(fn, types.FunctionType) and fn.__closure__:
            found = None
            for cell in fn.__closure__:
                try:
                    value = cell.cell_contents
                except ValueError:
                    continue
                if callable(value) and getattr(value, "__name__", "") == name and value is not fn:
                    found = value
                    break
            if found is None:
                break
            fn = found
        out[name] = fn
    return out

BLOCK = 5
REACHER_LINK = 0.12
REACHER_RADIUS = 0.01
REACHER_REACH = 0.24
CUBE_ACTION_SCALE = 0.05  # metres per unit normalised action, OGBench manipspace action_range
CUBE_TRACK = 0.40  # realised / commanded displacement per control step; fitted on host 2 (cube_scale_probe: slope 0.401, R^2 0.93, 240 steps)
CUBE_WORKSPACE = np.asarray([[0.25, -0.35, 0.02], [0.6, 0.35, 0.35]])
CUBE_RADIUS = 0.02
CUBE_HALF = 0.02  # pinned 4 cm cube, as geometry.recorded('cube')
CUBE_GRASP_DISTANCE = 0.04  # pinch site within this of the cube centre and gripper closing -> carried


def _fingertip(q):
    angles = np.array([q[0], q[0] + q[1]])
    return REACHER_LINK * np.array([np.cos(angles).sum(), np.sin(angles).sum()])


def _jacobian(q):
    s1, s12 = np.sin(q[0]), np.sin(q[0] + q[1])
    c1, c12 = np.cos(q[0]), np.cos(q[0] + q[1])
    return REACHER_LINK * np.array([[-s1 - s12, -s12], [c1 + c12, c12]])


def _unit(v, fallback):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else np.asarray(fallback, dtype=np.float64)


class ReacherOracle:
    task = "reacher"

    @staticmethod
    def _control_env(env):
        """Return (action_scale wrapper or None, dm_control control.Environment)."""
        inner = env.unwrapped.env
        wrapper = inner if hasattr(inner, "_transform") else None
        control = inner
        while not hasattr(control, "_n_sub_steps") and hasattr(control, "_env"):
            control = control._env
        if not hasattr(control, "_n_sub_steps"):
            raise ValueError("Reacher control environment not found")
        return wrapper, control

    def stepping(self, env):
        wrapper, control = self._control_env(env)
        transform = wrapper._transform if wrapper is not None else (lambda a: a)
        repeat = int(getattr(env.unwrapped, "action_repeat", 1))
        return transform, int(control._n_sub_steps), repeat

    def position(self, env):
        q = np.asarray(env.unwrapped.env.physics.data.qpos, dtype=np.float64)
        return _fingertip(q[:2])

    def rollout(self, env, hazard, blocks):
        blocks = np.asarray(blocks, dtype=np.float64)
        n = blocks.shape[0]
        live = env.unwrapped.env.physics
        transform, n_sub, repeat = self.stepping(env)
        raw = _raw_mujoco()
        legacy = bool(getattr(live, "legacy_step", False))
        centre = np.asarray(hazard["center"], dtype=np.float64)[:2] if hazard is not None else None
        rho = float(hazard["radius"]) if hazard is not None else 0.0
        clear = np.full(n, np.inf)
        safe_steps = np.full(n, BLOCK, dtype=np.int64)
        samples = BLOCK * repeat * n_sub + 1
        path = np.zeros((n, samples, 3))
        for i in range(n):
            scratch = live.copy(share_model=True)
            q_prev = np.asarray(scratch.data.qpos[:2], dtype=np.float64).copy()
            p_prev = _fingertip(q_prev)
            path[i, 0, :2] = p_prev
            s = 1
            for k in range(BLOCK):
                a = np.clip(blocks[i, k], -1.0, 1.0)
                for _ in range(repeat):
                    np.copyto(scratch.data.ctrl, transform(a))
                    for _ in range(n_sub):
                        # Mirror dm_control Physics.step(1) for this version, unpatched.
                        if legacy:
                            raw["mj_step2"](scratch.model.ptr, scratch.data.ptr)
                            raw["mj_step1"](scratch.model.ptr, scratch.data.ptr)
                        else:
                            raw["mj_step"](scratch.model.ptr, scratch.data.ptr, 1)
                        q = np.asarray(scratch.data.qpos[:2], dtype=np.float64).copy()
                        p = _fingertip(q)
                        path[i, s, :2] = p
                        s += 1
                        if centre is not None:
                            d = segment_distance(centre, p_prev, p) - REACHER_RADIUS - rho
                            d -= REACHER_REACH * np.abs(q - q_prev).sum() ** 2 / 8
                            if d <= 0 and safe_steps[i] == BLOCK:
                                safe_steps[i] = k
                            clear[i] = min(clear[i], d)
                        q_prev, p_prev = q, p
        return clear, safe_steps, path

    def directions(self, env, hazard, block):
        q = np.asarray(env.unwrapped.env.physics.data.qpos[:2], dtype=np.float64)
        pos = _fingertip(q)
        centre = np.asarray(hazard["center"], dtype=np.float64)[:2]
        e_rad = _unit(pos - centre, [1.0, 0.0])
        e_perp_c = np.array([-e_rad[1], e_rad[0]])
        jt = _jacobian(q).T
        e = _unit(jt @ e_rad, [1.0, 0.0])
        e_perp = _unit(jt @ e_perp_c, [-e[1], e[0]])
        d = _unit(np.asarray(block, dtype=np.float64).sum(axis=0), [0.0, 0.0])
        return e, e_perp, float(d @ e), float(d @ e_perp)


class CubeOracle:
    task = "cube"

    def __init__(self, track=CUBE_TRACK):
        self.track = float(track)

    def position(self, env):
        u = env.unwrapped
        return np.asarray(u._data.site_xpos[u._pinch_site_id], dtype=np.float64).copy()

    def box(self, env):
        """Live cube box (centre, rotation, half sizes) and whether it is carried by the gripper."""
        u = env.unwrapped
        qpos = np.asarray(u.data.qpos, dtype=np.float64)
        centre, rot = qpos[14:17].copy(), rotation(qpos[17:21])
        opening = float(np.clip(qpos[u._gripper_opening_joint_id] / 0.8, 0.0, 1.0))
        carried = (np.linalg.norm(centre - self.position(env)) < CUBE_GRASP_DISTANCE) and opening < 0.6
        return centre, rot, carried

    def rollout(self, env, hazard, blocks):
        """Kinematic pinch-site rollout; the cube box is static, or translated with the pinch site
        when it is carried. Clearance is the minimum over the ball term (swept) and the box term
        (sampled at step ends), matching the two terms of geometry.swept_clearance for Cube."""
        blocks = np.asarray(blocks, dtype=np.float64)
        n = blocks.shape[0]
        start = self.position(env)
        centre = np.asarray(hazard["center"], dtype=np.float64) if hazard is not None else None
        rho = float(hazard["radius"]) if hazard is not None else 0.0
        pos = np.repeat(start[None], n, axis=0)
        clear = np.full(n, np.inf)
        safe_steps = np.full(n, BLOCK, dtype=np.int64)
        path = np.zeros((n, BLOCK + 1, 3))
        path[:, 0] = pos
        box_c, box_r, carried = (None, None, False)
        if centre is not None:
            box_c, box_r, carried = self.box(env)
            half = np.full(3, CUBE_HALF)
            box_clear = shape_clearance({"balls": [], "polygons": [], "boxes": [(box_c, box_r, half)]}, hazard)
            clear[:] = box_clear
            if box_clear <= 0:
                safe_steps[:] = 0
        for k in range(BLOCK):
            a = np.clip(blocks[:, k, :3], -1.0, 1.0)
            target = np.clip(pos + a * CUBE_ACTION_SCALE, CUBE_WORKSPACE[0], CUBE_WORKSPACE[1])
            nxt = pos + self.track * (target - pos)
            if centre is not None:
                for i in range(n):
                    d = segment_distance(centre, pos[i], nxt[i]) - CUBE_RADIUS - rho
                    if carried:
                        moved = box_c + (nxt[i] - start)
                        d = min(d, shape_clearance({"balls": [], "polygons": [], "boxes": [(moved, box_r, half)]}, hazard))
                    if d <= 0 and safe_steps[i] == BLOCK:
                        safe_steps[i] = k
                    clear[i] = min(clear[i], d)
            pos = nxt
            path[:, k + 1] = pos
        return clear, safe_steps, path

    def directions(self, env, hazard, block):
        pos = self.position(env)
        centre = np.asarray(hazard["center"], dtype=np.float64)
        e_rad = _unit(pos - centre, [1.0, 0.0, 0.0])
        perp = np.cross(e_rad, [0.0, 0.0, 1.0])
        if np.linalg.norm(perp) < 1e-6:
            perp = np.cross(e_rad, [0.0, 1.0, 0.0])
        e_perp3 = _unit(perp, [0.0, 1.0, 0.0])
        dim = np.asarray(block).shape[-1]
        e = np.zeros(dim)
        e_perp = np.zeros(dim)
        e[:3], e_perp[:3] = e_rad, e_perp3
        d = _unit(np.asarray(block, dtype=np.float64).sum(axis=0), np.zeros(dim))
        return e, e_perp, float(d @ e), float(d @ e_perp)


def make_oracle(task):
    if task == "reacher":
        return ReacherOracle()
    if task == "cube":
        return CubeOracle()
    raise ValueError("No oracle for task %r" % (task,))
