"""Passive substep observations. No action veto, extra reset or physics step."""

import numpy as np

from .geometry import T_POLYGONS, clearance, recorded, rotation, swept_clearance, xyz

# Opt-in (graded study, 9 Nov 2026): PushT counts the pusher only, so the T block may pass through a hazard.
# Default False keeps every earlier study unchanged (the T polygons count).
PUSHER_ONLY = False


class PhysicalReader:
    def __init__(self, task, env):
        self.task, self.env = task, env.unwrapped
        self.scratch = None
        if task == "cube":
            import mujoco

            self.scratch = mujoco.MjData(self.env.model)
            model = self.env.model
            site = self.env._pinch_site_id
            body = int(model.site_bodyid[site])
            self.reach_bound = float(np.linalg.norm(model.site_pos[site]))
            ancestors = []
            while body:
                ancestors.append(body)
                self.reach_bound += float(np.linalg.norm(model.body_pos[body]))
                body = int(model.body_parentid[body])
            # The pinch site's kinematic chain contains the six arm joints only.
            chain_joints = [j for j in range(model.njnt) if model.jnt_bodyid[j] in ancestors]
            if sorted(int(model.jnt_qposadr[j]) for j in chain_joints) != list(range(6)):
                raise ValueError("Unexpected pinch-site kinematic chain")
        self.validate()

    def shape(self):
        env, task = self.env, self.task
        if task == "tworoom":
            return recorded(task, np.asarray(env._get_obs()))
        if task == "pusht":
            reference = xyz(np.asarray(env.agent.position))
            return {
                "reference": reference,
                "balls": [(reference, 15.0)],
                "boxes": [],
                "polygon_origin": np.asarray(env.block.local_to_world((0, 0))),
                "polygon_angle": float(env.block.angle),
                "polygons": [] if PUSHER_ONLY else [
                    np.array([env.block.local_to_world(v) for v in shape.get_vertices()])
                    for shape in env.block.shapes
                ],
            }
        if task == "reacher":
            return recorded(task, env.env.physics.data.qpos)
        import mujoco

        self.scratch.qpos[:] = env.data.qpos
        # Kinematics on separate scratch data cannot change the live solver state.
        mujoco.mj_kinematics(env.model, self.scratch)
        reference = self.scratch.site_xpos[env._pinch_site_id].copy()
        qpos = np.asarray(env.data.qpos)
        return {
            "reference": reference,
            "balls": [(reference, 0.02)],
            "polygons": [],
            "boxes": [(qpos[14:17].copy(), rotation(qpos[17:21]), np.full(3, 0.02))],
            "ball_joint_angles": qpos[:6].copy(),
            "ball_reach_bound": self.reach_bound,
        }

    def validate(self):
        env, task = self.env, self.task
        if task == "tworoom":
            radius = float(env.variation_space["agent"]["radius"].value.item())
            if radius != 7:
                raise ValueError("TwoRoom radius differs from frozen geometry")
        elif task == "pusht":
            if len(env.agent.shapes) != 1 or next(iter(env.agent.shapes)).radius != 15:
                raise ValueError("PushT pusher differs from frozen disk")
            vertices = [np.asarray(s.get_vertices()) for s in env.block.shapes]

            def sorted_vertices(p):
                return sorted(tuple(float(v) for v in row) for row in p)

            if sorted(map(sorted_vertices, vertices)) != sorted(map(sorted_vertices, T_POLYGONS)):
                raise ValueError("PushT block differs from frozen full T geometry")
        elif task == "reacher":
            physics = env.env.physics
            if not np.isclose(physics.named.model.geom_size["finger"][0], 0.01):
                raise ValueError("Reacher fingertip radius differs")
            observed = np.asarray(physics.named.data.geom_xpos["finger"][:2])
            expected = self.shape()["reference"][:2]
            if not np.allclose(observed, expected, atol=1e-7, rtol=0):
                raise ValueError("Reacher kinematics do not match the live fingertip")
        else:
            # The physical collision box must be the pinned 4 cm cube.
            gids = env._cube_geom_ids_list[0]
            if not any(np.allclose(env.model.geom_size[g], [0.02] * 3, atol=1e-10) for g in gids):
                raise ValueError("Cube collision geometry differs")
            if not np.allclose(
                self.shape()["reference"], env.compute_ob_info()["proprio/effector_pos"], atol=1e-6
            ):
                raise ValueError("Cube pinch-site coordinates differ")


class Monitor:
    def __init__(self, reader, hazard):
        self.reader, self.hazard = reader, hazard
        self.previous = reader.shape()
        self.sampled_min = clearance(self.previous, hazard)
        self.swept_min = self.sampled_min
        self.samples = 1
        self.points = [self.previous["reference"].copy()]
        self.pending = []
        self.patches = []
        self.last_swept = self.sampled_min

    def observe(self, *, controller_boundary=False):
        current = self.reader.shape()
        self.sampled_min = min(self.sampled_min, clearance(current, self.hazard))
        self.last_swept = swept_clearance(self.previous, current, self.hazard)
        self.swept_min = min(self.swept_min, self.last_swept)
        self.previous = current
        self.pending.append(current)
        self.samples += 1
        if controller_boundary:
            self.points.append(current["reference"].copy())

    def install(self):
        import mujoco
        import pymunk

        task = self.reader.task

        def patch(owner, name, function):
            self.patches.append((owner, name, getattr(owner, name)))
            setattr(owner, name, function)

        if task == "pusht":
            original = pymunk.Space.step

            def step(space, dt):
                result = original(space, dt)
                if space is self.reader.env.space:
                    self.observe()
                return result

            patch(pymunk.Space, "step", step)
        elif task in ("reacher", "cube"):
            original, original2 = mujoco.mj_step, mujoco.mj_step2

            def step(model, data, nstep=1):
                # The engineering parity test checks this against one native nstep call.
                for _ in range(nstep):
                    original(model, data, nstep=1)
                    self.observe()

            def step2(model, data):
                result = original2(model, data)
                self.observe()
                return result

            patch(mujoco, "mj_step", step)
            patch(mujoco, "mj_step2", step2)

    @property
    def violation(self):
        return bool(self.hazard is not None and self.swept_min <= 0)

    def close(self):
        for owner, name, original in reversed(self.patches):
            setattr(owner, name, original)
        self.patches.clear()

    def take_samples(self):
        samples, self.pending = self.pending, []
        return samples
