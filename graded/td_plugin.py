"""Graded study (9 Nov 2026): TD-MPC2 on Meta-World basketball, the plug-in for graded/core.py.

The frozen planner is the TD-MPC2 mw-basketball-1 checkpoint run exactly as the 8 Nov residual study ran it
(residual_tdmpc2: config.yaml, model size 5, eval_mode=True, torch.manual_seed(seed) per episode, action repeat 2,
100 decisions). The hazard is an axis-aligned cube on the gripper's (tcp_center) path, body radius 0
(hazard_design/DESIGN.md).

Kinds (graded/INTERFACES.md):
  native  the planner's own action; with log_senses, the 8 senses of the proposed action and of the warm-start
          action (MPPI's previous mean shifted forward) are logged per decision (None outside the outer limit).
  steer   core.SteerState on the proposed action. Body map: horizontal target displacement
          D' = factor * d + sideways * step * n, d = the predicted gripper displacement of the native action;
          the xy command changes by (D' - d) / 0.02 m per unit action, clipped to [-1, 1]. z and the gripper
          command a[3] are never touched (detours are horizontal only).
  choose  core.ChooseState inside MPPI: the gate reads the warm-start action (1 check); when open, the keyed
          iterations check the top-M sampled trajectories by planner value (kinematic check of the first action,
          safe share over the planning horizon), re-key them, and select the elites (and the MPPI weights) by key.
          Executed choice: the best-keyed checked candidate's first action if exec_keyed, else MPPI's own draw.
  stack   choose, then steer on the action choose's planner run produced.

Kinematic check (the "oracle"): Meta-World moves the hand mocap by action * 0.01 m per env step, clipped to the
mocap box, and a decision is 2 env steps (0.02 m per unit action per decision, verified exactly on the mocap). The
arm follows the mocap through a soft weld with a lag of about 3 decisions (measured on seeds 5000-5002: one-decision
gripper motion = 0.37 x lag + 0.12 x mocap step), so the checked point is the gripper's COMMANDED point,
eef + (new mocap - hand), i.e. where the gripper is being pulled. Its direction agrees with the gripper's actual
motion over the next 1 and 3 decisions with median cosine 0.975 / 0.978 (the bare 0.02 x action step: 0.78 / 0.93).
The body map is exact on the commanded point up to clipping; the physical one-decision gripper change is logged
separately (body_tcp1, from an exact MuJoCo lookahead on a scratch copy of the sim, which never touches the real run).
Off means native: with the trigger shut the original planner runs and its action object is executed untouched.
"""

import hashlib
import os
import time
import types

import numpy as np

from graded import core

SYSTEM = "mw-basketball"
MPU = 0.02  # metres of mocap travel per unit action per decision (2 env steps x action_scale 0.01)
EVAL_MODE = True  # as residual_tdmpc2 ran basketball (state.json eval_mode: true)
G = {}


# ------------------------------------------------------------------ set-up


def init(device):
    """Load the frozen agent and the basketball env once per worker process (call before any job)."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
    gl = os.environ.get("MUJOCO_GL_FORCE", "egl")
    os.environ["MUJOCO_GL"] = gl
    os.environ["PYOPENGL_PLATFORM"] = gl
    import torch

    torch.backends.cudnn.benchmark = False
    from residual_tdmpc2 import systems

    agent, cfg = systems.load_agent(SYSTEM)
    from common import math as tdmath  # tdmpc2/common/math.py (path set by load_agent)

    G.update(torch=torch, agent=agent, cfg=cfg, sys=systems.make_system(SYSTEM, hazard=None), systems=systems,
             tdmath=tdmath, native_plan=agent._plan, keyed_plan=types.MethodType(keyed_plan, agent), ctx=None)
    agent._plan_val = agent._plan  # the property `plan` returns this; native = the original bound method


# ------------------------------------------------------------------ geometry


class Cube:
    def __init__(self, case, goal):
        hz = case["hazard"]
        self.lo = np.asarray(hz["box_lo"], float)
        self.hi = np.asarray(hz["box_hi"], float)
        self.c = (self.lo + self.hi) / 2.0
        self.half3 = (self.hi - self.lo) / 2.0
        self.half = float(self.half3.mean())
        self.step = float(hz.get("step_m", case.get("step")))
        self.goal = np.asarray(goal, float)

    def clear(self, p):
        """Signed distance (m) of point(s) p (..., 3) to the cube; negative inside. Body radius 0."""
        q = np.abs(np.asarray(p, float) - self.c) - self.half3
        out = np.linalg.norm(np.maximum(q, 0.0), axis=-1)
        ins = np.minimum(q.max(axis=-1), 0.0)
        return out + ins


def mocap(s):
    return np.asarray(s.env.data.mocap_pos[0], float).copy()


def hand(s):
    return np.asarray(s.env.data.body("hand").xpos, float).copy()


def predict(s, a):
    """Commanded gripper point after action a: eef + (clip(mocap + 0.02 a_xyz) - hand)."""
    m1 = np.clip(mocap(s) + MPU * np.clip(np.asarray(a, float)[:3], -1.0, 1.0), s.low, s.high)
    return s.eef() + (m1 - hand(s))


def chain(s, pos, m0, h0, acts):
    """Commanded gripper points after each action of candidate trajectories. acts (H, M, 3) -> (H, M, 3)."""
    m = np.broadcast_to(m0, acts.shape[1:]).copy()
    pts = np.empty_like(acts, dtype=float)
    for k in range(acts.shape[0]):
        m = np.clip(m + MPU * np.clip(acts[k], -1.0, 1.0), s.low, s.high)
        pts[k] = pos + (m - h0)
    return pts


def lookahead(s, a):
    """Exact one-decision gripper point for action a (2 env steps) simulated on a scratch copy of the sim."""
    import mujoco

    env = s.env
    sc = G.get("scratch")
    if sc is None:
        sc = G["scratch"] = mujoco.MjData(env.model)
    mujoco.mj_copyData(sc, env.model, env.data)
    a = np.asarray(a, float)
    for _ in range(2):
        mp = sc.mocap_pos + (np.clip(a[:3], -1.0, 1.0) * env.action_scale)[None]
        mp[0, :] = np.clip(mp[0, :], env.mocap_low, env.mocap_high)
        sc.mocap_pos[:] = mp
        sc.mocap_quat[:] = np.array([1, 0, 1, 0])
        sc.ctrl[:] = [a[-1], -a[-1]]
        mujoco.mj_step(env.model, sc, nstep=env.frame_skip)
        mujoco.mj_rnePostConstraint(env.model, sc)
    return (np.asarray(sc.site("rightEndEffector").xpos) + np.asarray(sc.site("leftEndEffector").xpos)) / 2.0


def raw_for(geo, pos, p1, left, fallback):
    """core.planar_raw for a predicted next point p1 (horizontal frame; near-zero travel uses the fallback)."""
    d = np.asarray(p1, float) - pos
    dh = d[:2] if np.linalg.norm(d[:2]) > 1e-4 else np.zeros(2)
    return core.planar_raw(pos, dh, geo.c, geo.half, float(geo.clear(p1)), geo.goal, left, fallback)


def senses_or_none(raw, step):
    if not core.within_outer(raw, step):
        return None
    return core.sense_vector(dict(raw, memory=0.0), step).round(5).tolist()


# ------------------------------------------------------------------ Choose inside MPPI


def features(ctx, acts):
    """The 7 Choose features for checked candidates. acts: (H, M, 3) numpy, sorted by planner value (best first)."""
    s, geo = ctx["sys"], ctx["geo"]
    step = geo.step
    pts = chain(s, ctx["pos"], ctx["mocap"], ctx["hand"], acts)
    clear_all = geo.clear(pts)  # (H, M)
    clear1 = clear_all[0]
    safe = (clear_all > 0.0).mean(axis=0)
    lateral = -ctx["side_sign"] * ((pts[0][:, :2] - geo.c[:2]) @ ctx["n"]) / step
    m = acts.shape[1]
    rank = np.arange(m) / max(m - 1, 1)
    ctx["best_clear"] = max(ctx["best_clear"], float(clear1.max()))
    unit = lambda x: np.clip(np.asarray(x, float) / step, -core.OUTER, core.OUTER) / core.OUTER  # noqa: E731
    f = np.stack([
        unit(clear1), safe, np.clip(lateral, -core.OUTER, core.OUTER) / core.OUTER, rank,
        np.full(m, ctx["ahead"]), np.full(m, float(unit(ctx["best_clear"]))), np.full(m, ctx["left"]),
    ], axis=1)
    return f, clear1


def keyed_plan(self, obs, t0=False, eval_mode=False, task=None):
    """tdmpc2.TDMPC2._plan (commit 332f19e) with Choose's keyed elite selection and executed choice.

    With gamma = 0 and exec off it performs exactly the original operations (tested for bitwise parity).
    """
    torch = G["torch"]
    math = G["tdmath"]
    ctx = G["ctx"]
    with torch.no_grad():
        z = self.model.encode(obs, task)
        if self.cfg.num_pi_trajs > 0:
            pi_actions = torch.empty(self.cfg.horizon, self.cfg.num_pi_trajs, self.cfg.action_dim, device=self.device)
            _z = z.repeat(self.cfg.num_pi_trajs, 1)
            for t in range(self.cfg.horizon - 1):
                pi_actions[t], _ = self.model.pi(_z, task)
                _z = self.model.next(_z, pi_actions[t], task)
            pi_actions[-1], _ = self.model.pi(_z, task)

        z = z.repeat(self.cfg.num_samples, 1)
        mean = torch.zeros(self.cfg.horizon, self.cfg.action_dim, device=self.device)
        std = torch.full((self.cfg.horizon, self.cfg.action_dim), self.cfg.max_std, dtype=torch.float, device=self.device)
        if not t0:
            mean[:-1] = self._prev_mean[1:]
        actions = torch.empty(self.cfg.horizon, self.cfg.num_samples, self.cfg.action_dim, device=self.device)
        if self.cfg.num_pi_trajs > 0:
            actions[:, :self.cfg.num_pi_trajs] = pi_actions

        rounds = ctx["rounds"]
        state = ctx["state"]
        best = None
        for it in range(self.cfg.iterations):
            r = torch.randn(self.cfg.horizon, self.cfg.num_samples - self.cfg.num_pi_trajs, self.cfg.action_dim, device=std.device)
            actions_sample = mean.unsqueeze(1) + std.unsqueeze(1) * r
            actions_sample = actions_sample.clamp(-1, 1)
            actions[:, self.cfg.num_pi_trajs:] = actions_sample
            if self.cfg.multitask:
                actions = actions * self.model._action_masks[task]

            value = self._estimate_value(z, actions, task).nan_to_num(0)
            sel = value
            if it in rounds:
                m = int(rounds[it])
                v = value.squeeze(1)
                top = torch.topk(v, m, dim=0).indices
                vv = v.double().cpu().numpy()
                top_np = top.cpu().numpy()
                acts = actions[:, top, :3].float().cpu().numpy()
                mu, sd = float(vv.mean()), float(vv.std()) + 1e-9
                zz = -(vv[top_np] - mu) / sd
                feats, clear1 = features(ctx, acts)
                key = state.keys(zz, feats)
                bonus = (zz - key) * sd  # gamma * tanh(f) in value units; exactly 0 when gamma = 0
                sel = value.clone()
                sel[top, 0] = sel[top, 0] + torch.as_tensor(bonus, dtype=value.dtype, device=value.device)
                ctx["checks"] += m
                j = int(np.argmin(key))
                if best is None or key[j] < best[0]:
                    best = (float(key[j]), actions[:, int(top_np[j])].clone(), float(clear1[j]), it)
                ctx["rounds_log"].append({"it": it, "m": m, "key_best": float(key[j]), "clear_best": float(clear1[j]),
                                          "clear_top": float(clear1[0])})
            elite_idxs = torch.topk(sel.squeeze(1), self.cfg.num_elites, dim=0).indices
            elite_value, elite_actions = sel[elite_idxs], actions[:, elite_idxs]

            max_value = elite_value.max(0).values
            score = torch.exp(self.cfg.temperature * (elite_value - max_value))
            score = score / score.sum(0)
            mean = (score.unsqueeze(0) * elite_actions).sum(dim=1) / (score.sum(0) + 1e-9)
            std = ((score.unsqueeze(0) * (elite_actions - mean.unsqueeze(1)) ** 2).sum(dim=1) / (score.sum(0) + 1e-9)).sqrt()
            std = std.clamp(self.cfg.min_std, self.cfg.max_std)
            if self.cfg.multitask:
                mean = mean * self.model._action_masks[task]
                std = std * self.model._action_masks[task]

        rand_idx = math.gumbel_softmax_sample(score.squeeze(1))
        actions = torch.index_select(elite_actions, 1, rand_idx).squeeze(1)
        if getattr(state, "version", 1) == 2:
            # v2 safeguard: one extra check on MPPI's own draw; keep it if it already clears the margin, else
            # execute the best keyed plan only if that plan is predicted clearer.
            sys_, geo = ctx["sys"], ctx["geo"]
            pts = chain(sys_, ctx["pos"], ctx["mocap"], ctx["hand"], actions[:, None, :3].float().cpu().numpy())
            native_clear = float(geo.clear(pts)[0][0])
            ctx["checks"] += 1
            ctx["native_clear"] = native_clear
            if best is not None and native_clear < state.margin_steps * geo.step and best[2] > native_clear:
                actions = best[1]
                ctx["exec_from"] = best[3]
        elif state.exec_keyed and best is not None:
            actions = best[1]
            ctx["exec_from"] = best[3]
        a, std = actions[0], std[0]
        if not eval_mode:
            a = a + std * torch.randn(self.cfg.action_dim, device=std.device)
        self._prev_mean.copy_(mean)
        return a.clamp(-1, 1)


# ------------------------------------------------------------------ episode


def run(job):
    torch = G["torch"]
    agent, s = G["agent"], G["sys"]
    case = job["case"]
    kind = job["kind"]
    if kind not in ("native", "steer", "choose", "stack"):
        raise ValueError(f"unknown kind {kind}")
    hz = case.get("hazard")
    s.hazard = None if hz is None else {"box_lo": list(hz["box_lo"]), "box_hi": list(hz["box_hi"])}
    steer = core.SteerState(job["x_steer"]) if kind in ("steer", "stack") else None
    choose = core.ChooseState(job["x_choose"]) if kind in ("choose", "stack") else None
    force_open = bool(job.get("force_open"))  # dev test only: keyed planner path with the gate held open
    log = bool(job.get("log_senses"))
    record = bool(job.get("record"))
    veto = bool(job.get("veto"))
    seed = int(case["seed"])
    n_dec = s.decisions

    started = time.monotonic()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    obs = s.reset(seed)
    goal = case.get("goal") or np.asarray(s.env._target_pos, float).tolist()
    geo = Cube(case, goal) if hz is not None else None
    step = geo.step if geo is not None else float(case.get("step", 0.019))
    agent._plan_val = G["native_plan"]

    h = hashlib.sha256()
    ret = 0.0
    senses_prop, senses_warm, body, body_tcp1, dirs, look_err = [], [], [], [], [], []
    tcp, ball, acts_log, steer_log, choose_log = [s.eef().tolist()], [s.obj().tolist()], [], [], []
    steer_open = choose_open = 0
    checks_max = 0
    success_at = None
    fallback = None
    for t in range(n_dec):
        left = 1.0 - t / n_dec
        pos = s.eef().copy()
        m0 = mocap(s)
        if fallback is None:
            fallback = (geo.goal[:2] - pos[:2]) if geo is not None else np.array([1.0, 0.0])
        checks = 0
        warm_raw = None
        if geo is not None and (choose is not None or log):
            warm = np.zeros(4) if t == 0 else agent._prev_mean[1].double().cpu().numpy()
            warm_raw, _, warm_n = raw_for(geo, pos, predict(s, warm), left, fallback)
            if log:
                senses_warm.append(senses_or_none(warm_raw, step))
        elif log:
            senses_warm.append(None)
        if choose is not None:
            agent._plan_val = G["native_plan"]
            if geo is not None:
                is_open, crec = choose.gate(warm_raw, step)
                checks += 1
                if force_open:
                    is_open = True
                if is_open:
                    choose_open += 1
                    G["ctx"] = dict(sys=s, geo=geo, pos=pos, mocap=m0, hand=hand(s), n=warm_n, left=left,
                                    side_sign=1.0 if warm_raw["side"] >= 0 else -1.0,
                                    ahead=float(np.clip(warm_raw["ahead"] / step, -core.OUTER, core.OUTER) / core.OUTER),
                                    best_clear=float(geo.clear(predict(s, warm))), state=choose,
                                    rounds=choose.checks_per_round(G["cfg"].iterations), checks=0, rounds_log=[],
                                    exec_from=None)
                    agent._plan_val = G["keyed_plan"]
                if record:
                    choose_log.append(dict(t=t, **crec))
        a = agent.act(torch.from_numpy(obs), t0=(t == 0), eval_mode=EVAL_MODE).numpy()
        if agent._plan_val is not G["native_plan"]:
            checks += G["ctx"]["checks"]
            if record:
                choose_log[-1].update(rounds=G["ctx"]["rounds_log"], exec_from=G["ctx"]["exec_from"])
            agent._plan_val = G["native_plan"]
            G["ctx"] = None
        proposed = a
        p1 = predict(s, a) if geo is not None else None
        d = (p1 - pos) if geo is not None else None
        if geo is not None and log:
            raw, _, _ = raw_for(geo, pos, p1, left, fallback)
            senses_prop.append(senses_or_none(raw, step))
        elif log:
            senses_prop.append(None)
        cmd = None
        tcp1_native = None
        if steer is not None and geo is not None:
            raw, _, n = raw_for(geo, pos, p1, left, fallback)
            act, factor, sideways, srec = steer.decide(raw, step)
            checks += 1
            if act:
                steer_open += 1
                target = factor * d[:2] + sideways * step * n
                a2 = np.array(a, dtype=np.float64, copy=True)
                a2[:2] = np.clip(a2[:2] + (target - d[:2]) / MPU, -1.0, 1.0)
                a2 = a2.astype(a.dtype)
                if veto and float(geo.clear(predict(s, a2))) < float(geo.clear(p1)):
                    srec["veto"] = 1
                else:
                    cmd = (target - d[:2]).copy()
                    realised = (predict(s, a2) - p1)[:2]
                    tcp1_native = lookahead(s, a)  # counterfactual: the native action, on a scratch copy
                    a = a2
            if record:
                steer_log.append(dict(t=t, **srec))
        if log and kind == "native" and t % 10 == 5:
            tcp1_native = lookahead(s, a)  # exactness check of the scratch lookahead on the real step
        checks_max = max(checks_max, checks)
        h.update(np.ascontiguousarray(a, dtype=np.float32).tobytes())
        acts_log.append(a)
        obs, r, info = s.step(a)
        ret += r
        new = s.eef()
        if success_at is None and float(info.get("success", 0.0)) >= 1.0:
            success_at = t + 1
        if cmd is not None:
            body.append([round(float(v), 6) for v in (*cmd, *realised)])
            body_tcp1.append([round(float(v), 6) for v in (new - tcp1_native)[:2]])
        elif tcp1_native is not None:
            look_err.append(float(np.abs(new - tcp1_native).max()))
        tcp.append(new.tolist())
        if geo is not None:
            dirs.append(d[:2].copy())
            hd = new[:2] - pos[:2]
            if np.linalg.norm(hd) > 1e-4:
                fallback = hd
        if record:
            ball.append(s.obj().tolist())

    out = s.outcome(ret)
    mins = out.get("min_margin_m")
    ball_end = s.obj()
    out_goal = np.asarray(goal, float)
    res = {
        "job_id": job["job_id"], "env": "basketball", "case_id": case.get("id"), "kind": kind, "completed": True,
        "success": bool(out["success"]), "violation": bool(out["violation"]),
        "safe_completion": bool(out["success"] and not out["violation"]),
        "min_clear_steps": (float(mins) / step) if mins is not None else None,
        "final_goal_steps": float(np.linalg.norm(ball_end - out_goal)) / step,
        "final_gripper_goal_steps": float(np.linalg.norm(s.eef() - out_goal)) / step,
        "success_final": bool(out.get("success_final")), "success_at": success_at,
        "steer_open": int(steer_open), "choose_open": int(choose_open),
        "decisions": int(n_dec), "checks_max": int(checks_max),
        "parity_sha256": h.hexdigest(), "actions": len(acts_log), "seconds": time.monotonic() - started,
        "seed": seed, "ret": float(ret), "veto": veto,
    }
    if log:
        res["senses_prop"] = senses_prop
        res["senses_warm"] = senses_warm
    if steer is not None:
        res["body"] = body  # [requested dx, dy, realised dx, dy] of the commanded gripper point (m)
        res["body_tcp1"] = body_tcp1  # actual one-decision gripper change vs the native action (exact lookahead), m
    if log and dirs:
        P = np.asarray(tcp)
        cos = []
        for k, dd in enumerate(dirs[:-3]):
            mv = P[k + 3, :2] - P[k, :2]
            if np.linalg.norm(dd) > 1e-3 and np.linalg.norm(mv) > 1e-3:
                cos.append(float(dd @ mv / np.linalg.norm(dd) / np.linalg.norm(mv)))
        res["kin_check"] = {"cmdpoint_cos3_median": round(float(np.median(cos)), 4) if cos else None,
                            "lookahead_max_err_m": max(look_err) if look_err else None, "lookahead_n": len(look_err)}
    if record:
        res["trace"] = {"tcp": tcp, "ball": ball, "samples": [x["eef"] for x in s.samples], "target": out["target"],
                        "goal": list(map(float, out_goal)), "success_at": success_at,
                        "actions": [[round(float(v), 6) for v in x] for x in acts_log],
                        "steer": steer_log, "choose": choose_log}
    return res
