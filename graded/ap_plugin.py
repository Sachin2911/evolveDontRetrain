"""Graded study (9 Nov 2026): the LeWM + Anchored Planning plug-in (TwoRoom, PushT, Reacher).

Supplies raw geometry in env units to `graded.core` and maps the adapter's outputs back to actions.

Hooks (installed once per worker process, all inert unless a job turns them on):
  - `planners.cem` is wrapped. With Choose shut (or no Choose) the ORIGINAL cem runs untouched. Native runs with
    `log_senses` only read the warm-start block (one oracle rollout, no model or RNG calls). With Choose open a
    replica of the native CEM loop (same draws, same 31 predictor calls) re-keys the top-M candidates of the keyed
    rounds and selects that round's elites by the key.
  - `TargetGaussian.choose` is wrapped. After the planner returns its 5-action block, Steer reads the senses of
    that block (one oracle check) and, with its trigger open, replaces the block by one Newton step of the body map.
    With the trigger shut the planner's own block object is returned.
  - `environments._make_live` is wrapped to capture the live env and log the body position after every env step.

Bodies: TwoRoom agent (7 px), PushT pusher only (15 px; the T never counts), Reacher fingertip (1 cm).
"""

import numpy as np
import torch

from graded import core
from qd_adapters import v4hook

BLOCK, SAMPLES, ITERATIONS, ELITES = 5, 300, 30, 30
SIGMA, STD_FLOOR = 1.0 / 3.0, 1e-5
FD_EPS = 0.05  # action units, central finite difference for the body map Jacobian
NEAR_ZERO = 0.1  # steps: a block displacement shorter than this has no travel direction


# ------------------------------------------------------------------ oracles (env units)


class Body:
    """Oracle rollouts of candidate blocks for the counted body, without touching the live env."""

    def __init__(self, task):
        self.task = task
        if task == "reacher":
            from residual.oracles_extra import ReacherOracle

            self.reacher = ReacherOracle()

    def position(self, env):
        if self.task == "tworoom":
            return np.asarray(env.unwrapped.agent_position.numpy(), dtype=np.float64)[:2].copy()
        if self.task == "pusht":
            return np.asarray(env.unwrapped.agent.position, dtype=np.float64)[:2].copy()
        return np.asarray(self.reacher.position(env), dtype=np.float64)[:2].copy()

    def rollout(self, env, hazard, blocks):
        """blocks (N, 5, 2) raw -> (clear[N], safe_steps[N], end displacement[N, 2]). Clearance in env units."""
        blocks = np.asarray(blocks, dtype=np.float64)
        if self.task == "tworoom":
            start = np.asarray(env.unwrapped.agent_position.numpy(), dtype=np.float64)
            clear, steps, path = v4hook.tworoom_rollout(env, start, blocks, hazard)
        elif self.task == "pusht":
            u = env.unwrapped
            clear, steps, path = v4hook.pusht_rollout(
                env, np.asarray(u.agent.position), np.asarray(u.agent.velocity), blocks, hazard
            )
        else:
            # mujoco.rollout from the live state on scratch MjData (no Physics copies; residual's per-candidate
            # Physics.copy oracle hung a worker under Choose). Same end positions as live stepping; clearance
            # omits the monitor's chord-error term (<= 0.5 mm on a 29-candidate check).
            clear, steps, path = v4hook.reacher_rollout(env, blocks, hazard)
        path = np.asarray(path, dtype=np.float64)
        disp = path[:, -1, :2] - path[:, 0, :2]
        return np.asarray(clear, dtype=np.float64), np.asarray(steps), disp


# ------------------------------------------------------------------ per-job state

CTX = {"active": False}


def begin(task, job, case, body, true_goal_fn=None):
    """Reset the hook state for one job."""
    kind = job["kind"]
    ap_case = case["ap_case"]
    hazard = ap_case.get("hazard")
    CTX.clear()
    CTX.update(
        active=True, task=task, kind=kind, body=body, env=None, hazard=hazard,
        step=float(case["step"]), body_radius=float(case["body_radius"]),
        goal=None if case.get("goal") is None else np.asarray(case["goal"], dtype=np.float64)[:2],
        cap=int(ap_case["action_cap"]), actions=0, positions=[], frames=[], record=bool(job.get("record")),
        log_senses=bool(job.get("log_senses")) and kind == "native",
        veto=bool(job.get("veto")),
        steer=core.SteerState(job["x_steer"]) if kind in ("steer", "stack") else None,
        choose=core.ChooseState(job["x_choose"]) if kind in ("choose", "stack") else None,
        decisions=0, senses_prop=[], senses_warm=[], body_log=[], checks=[], vetoed=0, steer_records=[],
        choose_records=[], last_dir=None, dec=None, true_goal=None,
    )


def end():
    CTX["active"] = False


def _geometry_ready():
    return CTX.get("active") and CTX.get("hazard") is not None and CTX.get("env") is not None


def _goal(pos):
    g = CTX["goal"]
    if g is None:
        g = CTX.get("true_goal")
    return pos if g is None else g


def _fallback(pos):
    if CTX["last_dir"] is not None:
        return CTX["last_dir"]
    to_goal = _goal(pos) - pos
    return to_goal if np.linalg.norm(to_goal) > 1e-9 else np.array([1.0, 0.0])


def _raw_senses(pos, disp, clear):
    """Raw senses in env units. A displacement under NEAR_ZERO steps carries no direction: the frame falls back to
    the previous decision's travel direction (the goal direction at the first decision)."""
    if np.linalg.norm(np.asarray(disp, dtype=np.float64)[:2]) < NEAR_ZERO * CTX["step"]:
        disp = np.zeros(2)
    hz = CTX["hazard"]
    size = float(hz["radius"]) + CTX["body_radius"]
    left = max(0.0, (CTX["cap"] - CTX["actions"]) / CTX["cap"])
    return core.planar_raw(pos, disp, hz["center"], size, clear, _goal(pos), left, _fallback(pos))


def _logged(raw):
    """Senses for the log: null outside the outer limit (memory 0, as on a native run)."""
    if not core.within_outer(raw, CTX["step"]):
        return None
    return core.sense_vector(dict(raw, memory=0.0), CTX["step"]).tolist()


def _to_raw(native, normalized, raw_bounds):
    """(N, 5*d) normalized -> (N, 5, d) raw actions, as the planner converts its choice."""
    normalized = np.asarray(normalized, dtype=np.float32).reshape(-1, BLOCK, native.action_dim)
    flat = normalized.reshape(-1, native.action_dim)
    raw = native.scaler.inverse_transform(flat).astype(np.float32)
    raw = np.clip(raw, raw_bounds[0], raw_bounds[1])
    return raw.reshape(normalized.shape[0], BLOCK, native.action_dim)


# ------------------------------------------------------------------ Steer body map


def steer_block(env, block, d, factor, sideways, n, low, high):
    """One Newton step: uniform 2-D action offset on the block so the body's block displacement becomes
    D' = factor * d + sideways * step * n. Returns (corrected block, commanded change, achieved change, clear)."""
    body, hazard, step = CTX["body"], CTX["hazard"], CTX["step"]
    block = np.asarray(block, dtype=np.float64)
    target = factor * d + sideways * step * n
    need = target - d
    e = np.eye(2) * FD_EPS
    probes = np.stack([np.clip(block + s * e[j][None, :], low, high) for j in range(2) for s in (1.0, -1.0)])
    _, _, disp = body.rollout(env, hazard, probes)
    jac = np.stack([(disp[0] - disp[1]) / (2 * FD_EPS), (disp[2] - disp[3]) / (2 * FD_EPS)], axis=1)
    delta = np.linalg.lstsq(jac, need, rcond=1e-6)[0]
    corrected = np.clip(block + delta[None, :], low, high)
    clear_c, steps_c, disp_c = body.rollout(env, hazard, corrected[None])
    return corrected, need, disp_c[0] - d, float(clear_c[0]), int(steps_c[0]), delta


# ------------------------------------------------------------------ the CEM wrapper (warm start, Choose)

_NATIVE = {}


def graded_cem(native, rgb, current, target, generator, normalized_prior, normalized_bounds, raw_bounds):
    dec = CTX.get("dec") if CTX.get("active") else None
    if dec is None or not _geometry_ready() or not (CTX["log_senses"] or CTX["choose"] is not None):
        return _NATIVE["cem"](native, rgb, current, target, generator, normalized_prior, normalized_bounds, raw_bounds)
    env, body = CTX["env"], CTX["body"]
    pos = body.position(env)
    warm = _to_raw(native, np.asarray(normalized_prior).reshape(1, -1), raw_bounds)
    clear_w, steps_w, disp_w = body.rollout(env, CTX["hazard"], warm)
    # AP's warm start is the fixed standardized-zero prior (planners.TargetGaussian), not a shifted plan, so its
    # small displacement carries no travel direction: its frame is the previous decision's travel direction.
    raw_w, t_w, n_w = _raw_senses(pos, np.zeros(2), clear_w[0])
    dec["warm"] = raw_w
    if CTX["log_senses"]:
        dec["senses_warm"] = _logged(raw_w)
    choose = CTX["choose"]
    if choose is None:
        return _NATIVE["cem"](native, rgb, current, target, generator, normalized_prior, normalized_bounds, raw_bounds)
    dec["checks"] = 1
    is_open, rec = choose.gate(raw_w, CTX["step"])
    dec["choose"] = rec
    if not is_open:
        return _NATIVE["cem"](native, rgb, current, target, generator, normalized_prior, normalized_bounds, raw_bounds)
    s_w = core.sense_vector(dict(raw_w, memory=0.0), CTX["step"])
    frame = dict(pos=pos, t=t_w, n=n_w, side=raw_w["side"], ahead=s_w[0], left=s_w[6], best=float(clear_w[0]))
    return keyed_cem(native, rgb, current, target, generator, normalized_prior, normalized_bounds, raw_bounds,
                     choose, frame, dec)


def keyed_cem(native, rgb, current, target, generator, normalized_prior, normalized_bounds, raw_bounds,
              choose, frame, dec):
    """Replica of the native CEM (qd_adapters.v4hook.patched_cem, inactive path) with keyed elite selection in the
    keyed rounds. Same draws, same 31 predictor calls, same final prior-vs-mean comparison."""
    env, body, hazard, step = CTX["env"], CTX["body"], CTX["hazard"], CTX["step"]
    prior = torch.as_tensor(normalized_prior, dtype=torch.float32, device=native.device).reshape(
        1, 1, BLOCK * native.action_dim).clone()
    lower = torch.as_tensor(np.tile(normalized_bounds[0], BLOCK), dtype=torch.float32, device=native.device)
    upper = torch.as_tensor(np.tile(normalized_bounds[1], BLOCK), dtype=torch.float32, device=native.device)
    projected_components = torch.zeros((), dtype=torch.int64, device=native.device)
    projected_blocks = torch.zeros((), dtype=torch.int64, device=native.device)
    scored_min = torch.full_like(lower, float("inf"))
    scored_max = torch.full_like(upper, float("-inf"))

    def project(value):
        return torch.minimum(torch.maximum(value, lower), upper)

    assert torch.equal(project(prior), prior), "Recorded prior is outside the action domain."
    mean = prior.clone()
    std = torch.full_like(mean, SIGMA)
    pixels = native.pixels(rgb)
    work_calls = work_transitions = 0

    def cost(actions):
        nonlocal work_calls, work_transitions
        bounded = project(actions)
        changed = bounded != actions
        projected_components.add_(changed.sum())
        projected_blocks.add_(changed.any(dim=-1).sum())
        actions = bounded
        scored_min.copy_(torch.minimum(scored_min, actions.amin(dim=(0, 1, 2))))
        scored_max.copy_(torch.maximum(scored_max, actions.amax(dim=(0, 1, 2))))
        count = actions.shape[1]
        info = dict(
            pixels=pixels[:, None].expand(-1, count, -1, -1, -1, -1),
            current_latent=current[:, None, None, :].expand(-1, count, -1, -1),
            subgoal_latent=target[:, None, :].expand(-1, count, -1),
        )
        values = native.latent_cost.get_cost(info, actions)
        assert values.shape == (1, count) and torch.isfinite(values).all()
        work_calls += 1
        work_transitions += count
        return values, bounded

    per_round = choose.checks_per_round(ITERATIONS)
    sgn = 1.0 if frame["side"] >= 0 else -1.0  # hazard on the left of travel -> 'away' is to the right
    centre = np.asarray(hazard["center"], dtype=np.float64)[:2]
    best = None  # (key, normalized block, round)
    checks = 1
    first_noise = None
    keyed_log = []
    for iteration in range(ITERATIONS):
        noise = torch.randn(1, SAMPLES, 1, BLOCK * native.action_dim, generator=generator, device=native.device)
        if iteration == 0:
            first_noise = noise[0, 1, 0].cpu().tolist()
            if getattr(native, "capture_initial_noise", False):
                native.initial_standard_noise = noise.detach().cpu().numpy().copy()
        candidates = noise * std[:, None] + mean[:, None]
        candidates[:, 0] = mean
        values, bounded = cost(candidates)
        m = per_round.get(iteration, 0)
        if m > 0:
            v = values[0].detach().cpu().numpy().astype(np.float64)
            z = (v - v.mean()) / (v.std() + 1e-9)
            top = np.argsort(v, kind="stable")[:m]
            norm_top = bounded[0, top, 0, :].detach().cpu().numpy()
            raw_top = _to_raw(native, norm_top, raw_bounds)
            clear, safe_steps, disp = body.rollout(env, hazard, raw_top)
            checks += m
            frame["best"] = max(frame["best"], float(np.max(clear)))
            end_pt = frame["pos"][None, :] + disp
            lateral = -sgn * ((end_pt - centre[None, :]) @ frame["n"]) / step
            feats = np.stack([
                np.clip(clear / step, -core.OUTER, core.OUTER) / core.OUTER,
                np.asarray(safe_steps, dtype=np.float64) / BLOCK,
                np.clip(lateral, -core.OUTER, core.OUTER) / core.OUTER,
                np.arange(m, dtype=np.float64) / max(m - 1, 1),
                np.full(m, frame["ahead"]),
                np.full(m, np.clip(frame["best"] / step, -core.OUTER, core.OUTER) / core.OUTER),
                np.full(m, frame["left"]),
            ], axis=1)
            key = z.copy()
            kt = choose.keys(z[top], feats)
            key[top] = kt
            elite_idx = np.argsort(key, kind="stable")[:ELITES]
            elite = torch.as_tensor(elite_idx[None, :], dtype=torch.int64, device=native.device)
            j = int(np.argmin(kt))
            if best is None or kt[j] < best[0]:
                best = (float(kt[j]), norm_top[j].copy(), iteration, float(clear[j]))
            keyed_log.append(dict(round=iteration, m=m, min_key=float(kt[j]), best_clear=float(np.max(clear)),
                                  elite_changed=int(len(set(elite_idx.tolist()) - set(np.argsort(v, kind="stable")[:ELITES].tolist())))))
        else:
            elite = torch.topk(values, ELITES, dim=1, largest=False).indices
        elites = candidates[torch.arange(1, device=native.device)[:, None], elite]
        mean = elites.mean(dim=1)
        std = elites.std(dim=1).clamp_min(STD_FLOOR)
    values, effective = cost(torch.stack((prior, mean), dim=1))
    improved = bool((values[0, 1] < values[0, 0]).item())
    chosen = effective[:, 1] if improved else effective[:, 0]
    assert work_calls == 31 and work_transitions == 9002
    normalized = chosen[0, 0].cpu().numpy().reshape(BLOCK, native.action_dim)
    keep_flag = improved
    executed = "mean" if improved else "prior"
    native_clear = None
    if getattr(choose, "version", 1) == 2:
        # v2 safeguard: one extra check on the planner's own block; keep it if it already clears the margin, else
        # execute the best keyed plan only if that plan is predicted clearer.
        nc, _, _ = body.rollout(env, hazard, _to_raw(native, normalized.reshape(1, -1), raw_bounds))
        native_clear = float(nc[0])
        checks += 1
        if best is not None and native_clear < choose.margin_steps * step and best[3] > native_clear:
            normalized = np.asarray(best[1], dtype=np.float32).reshape(BLOCK, native.action_dim)
            keep_flag = True
            executed = "keyed-r%d" % best[2]
        else:
            executed = executed + ("-clear" if native_clear >= choose.margin_steps * step else "-kept")
    elif choose.exec_keyed and best is not None:
        normalized = np.asarray(best[1], dtype=np.float32).reshape(BLOCK, native.action_dim)
        keep_flag = True
        executed = "keyed-r%d" % best[2]
    dec["checks"] = checks
    dec["choose"].update(executed=executed, rounds=keyed_log, native_clear=native_clear)
    raw = native.scaler.inverse_transform(normalized).astype(np.float32)
    raw = np.clip(raw, raw_bounds[0], raw_bounds[1]).astype(np.float32)
    roundtrip_error = float(np.max(np.abs(native.scaler.transform(raw) - normalized)))
    assert raw.shape == (BLOCK, native.action_dim) and np.isfinite(raw).all()
    detail = dict(
        normalized_prior=prior[0, 0].cpu().tolist(),
        normalized_selected=normalized.tolist(), refined_mean_selected=keep_flag,
        prior_cost=float(values[0, 0]), refined_cost=float(values[0, 1]),
        selected_cost=float(values.min()), first_standard_noise=first_noise,
        predictor_batch_calls=work_calls, predictor_candidate_transitions=work_transitions,
        projected_scored_blocks=int(projected_blocks.item()),
        projected_scored_components=int(projected_components.item()),
        scored_normalized_min=scored_min.cpu().tolist(),
        scored_normalized_max=scored_max.cpu().tolist(),
        normalized_execution_roundtrip_error=roundtrip_error,
    )
    detail["graded_choose"] = {"executed": executed, "checks": checks}
    return raw, detail


# ------------------------------------------------------------------ the choose wrapper (proposal senses, Steer)


def after_planner(proposed, detail):
    """Called with the planner's block (after Choose, if any). Returns the block to execute (same object if shut)."""
    dec = CTX["dec"]
    steer = CTX["steer"]
    if not _geometry_ready() or not (CTX["log_senses"] or steer is not None or CTX["choose"] is not None):
        return proposed, detail
    env, body = CTX["env"], CTX["body"]
    pos = body.position(env)
    block = np.asarray(proposed)
    clear_p, steps_p, disp_p = body.rollout(env, CTX["hazard"], block[None])
    raw_p, t_p, n_p = _raw_senses(pos, disp_p[0], clear_p[0])
    CTX["last_dir"] = t_p  # the frame's fallback direction at the next decision (bookkeeping, not a check)
    if CTX["log_senses"]:
        dec["senses_prop"] = _logged(raw_p)
    if steer is None:
        return proposed, detail
    dec["checks"] = dec.get("checks", 0) + 1
    act, factor, sideways, rec = steer.decide(raw_p, CTX["step"])
    dec["steer"] = rec
    if not act:
        return proposed, detail
    low, high = env.action_space.low, env.action_space.high
    corrected, need, got, clear_c, steps_c, delta = steer_block(env, block, disp_p[0], factor, sideways, n_p, low, high)
    rec.update(clear_native=float(clear_p[0]), clear_corrected=clear_c, delta=delta.tolist())
    CTX["body_log"].append([float(need[0]), float(need[1]), float(got[0]), float(got[1])])
    if CTX["veto"] and clear_c < float(clear_p[0]):
        rec["vetoed"] = 1
        CTX["vetoed"] += 1
        return proposed, detail
    out = corrected.astype(block.dtype)
    if isinstance(detail, dict):
        detail = dict(detail, graded_steer={k: v for k, v in rec.items() if k != "senses"})
    return out, detail


def install(runner):
    """Install the hooks once per process. Inert until `begin` activates them for a job."""
    envs = runner.envs
    original_make = envs._make_live

    def make_live(*args, **kwargs):
        out = original_make(*args, **kwargs)
        env = out[0]
        if not CTX.get("active"):
            return out
        CTX["env"] = env
        body = CTX["body"]
        try:
            from qd_adapters.geometry import recorded

            CTX["true_goal"] = np.asarray(recorded(CTX["task"], out[3])["reference"], dtype=np.float64)[:2]
        except Exception:
            CTX["true_goal"] = None
        CTX["positions"].append(body.position(env).tolist())
        if CTX["record"]:
            try:
                CTX["frames"].append(np.ascontiguousarray(env.render()))
            except Exception:
                pass
        original_step = env.step

        def step(action):
            result = original_step(action)
            CTX["actions"] += 1
            CTX["positions"].append(body.position(env).tolist())
            if CTX["record"]:
                try:
                    CTX["frames"].append(np.ascontiguousarray(env.render()))
                except Exception:
                    pass
            return result

        env.step = step
        return out

    envs._make_live = make_live

    from anchored_planning import planners
    from anchored_planning.planners import TargetGaussian

    if not _NATIVE:
        _NATIVE["cem"] = planners.cem
        planners.cem = graded_cem
    original_choose = TargetGaussian.choose

    def choose(self, *args, **kwargs):
        if not CTX.get("active"):
            return original_choose(self, *args, **kwargs)
        dec = {"senses_prop": None, "senses_warm": None, "checks": 0}
        CTX["dec"] = dec
        proposed, detail = original_choose(self, *args, **kwargs)
        proposed, detail = after_planner(proposed, detail)
        CTX["decisions"] += 1
        CTX["senses_warm"].append(dec["senses_warm"])
        CTX["senses_prop"].append(dec["senses_prop"])
        CTX["checks"].append(dec["checks"])
        if "steer" in dec:
            CTX["steer_records"].append(dec["steer"])
        if "choose" in dec:
            CTX["choose_records"].append({k: v for k, v in dec["choose"].items() if k != "senses"})
        CTX["dec"] = None
        return proposed, detail

    TargetGaussian.choose = choose
