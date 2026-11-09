"""Recipe v4 hook: kinematic margin oracle + lexicographic key at the CEM elite sort.

The patched `cem` reproduces the native update exactly when no rule is active
(same RNG draws, same predictor calls, same elite selection, same final choice)
and records the final candidate population for the offline pool. With a rule
active, each iteration's elite set is chosen by the key (-tier, native cost)
where tier = safe replan steps of the candidate block under the oracle, and
the executed block is chosen among {prior, refined mean, best-keyed elite}.
"""

import numpy as np
import torch

from .geometry import polygon_distance

BLOCK, SAMPLES, ITERATIONS, ELITES = 5, 300, 30, 30
SIGMA, STD_FLOOR = 1.0 / 3.0, 1e-5
BIG = 1.0e6

CONTEXT = {"active": False}

# Gated gain residual (the "nudge"). Gate: the executed block is predicted to intrude, or (Reacher) to pass
# closer than the margin; the margin covers the measured gap between the oracle's straight-segment sweep and
# the monitor's swept test (sort-arm grazes of 1e-5..1.2e-4 m). Gate open -> try ref + g (block - ref) for
# each gain g (largest first; negative = reversed torque for Reacher, backing off for the point agents) and
# execute the first the oracle clears. Gate closed -> nothing changes, bit for bit.
GAINS = (0.9, 0.75, 0.6, 0.45, 0.3, 0.15, 0.0, -0.25, -0.5, -1.0)
NUDGE_MARGIN = {"reacher": 0.002, "tworoom": 0.0, "pusht": 0.0}


def begin_decision(*, task, rule, hazard, env, step, rng, genotype=None):
    CONTEXT.clear()
    CONTEXT.update(
        active=True, task=task, rule=rule, hazard=hazard, env=env, step=step, rng=rng, record=None,
        genotype=genotype,
    )


def take_record():
    record = CONTEXT.get("record")
    CONTEXT.clear()
    CONTEXT["active"] = False
    return record


# ---------------------------------------------------------------- kinematics


def _segment_clearance(p0, p1, center, body_radius, hazard_radius):
    """Vectorised swept clearance of a disc moving p0->p1 against a disc hazard."""
    d = p1 - p0
    rel = center[None, :2] - p0
    dd = np.einsum("ij,ij->i", d, d)
    t = np.clip(np.einsum("ij,ij->i", rel, d) / np.maximum(dd, 1e-30), 0.0, 1.0)
    closest = p0 + t[:, None] * d
    dist = np.linalg.norm(closest - center[None, :2], axis=1)
    return dist - body_radius - hazard_radius


def tworoom_rollout(env, start, actions, hazard):
    """Exact TwoRoom dynamics for N candidate blocks. actions: (N, 5, 2) raw in [-1, 1]."""
    u = env.unwrapped
    speed = float(u.variation_space["agent"]["speed"].value.item())
    r = float(u.variation_space["agent"]["radius"].value.item())
    bs, size, c = float(u.BORDER_SIZE), float(u.IMG_SIZE), float(u.WALL_CENTER)
    half = u.wall_thickness // 2
    axis = int(u.wall_axis)
    doors = [
        (float(u.door_positions[i]), float(u.door_sizes[i]) + 1.75) for i in range(u.num_doors)
    ]
    n = actions.shape[0]
    pos = np.repeat(np.asarray(start, dtype=np.float64)[None, :2], n, axis=0)
    clear = np.full(n, np.inf)
    safe_steps = np.full(n, BLOCK, dtype=np.int64)
    path = np.zeros((n, BLOCK + 1, 2))
    path[:, 0] = pos
    for k in range(BLOCK):
        a = np.clip(actions[:, k, :].astype(np.float64), -1.0, 1.0)
        nxt = pos + a * speed
        nxt[:, 0] = np.clip(nxt[:, 0], bs + r, size - bs - r)
        nxt[:, 1] = np.clip(nxt[:, 1], bs + r, size - bs - r)
        move, side = (0, 1) if axis == 1 else (1, 0)
        started_low = pos[:, move] < c
        low_edge, high_edge = c - half - r, c + half + r
        coord = nxt[:, side]
        in_door = np.zeros(n, dtype=bool)
        for centre, margin in doors:
            in_door |= (coord >= centre - margin) & (coord <= centre + margin)
        hit_low = started_low & (nxt[:, move] > low_edge) & ~in_door
        hit_high = ~started_low & (nxt[:, move] < high_edge) & ~in_door
        nxt[hit_low, move] = low_edge - 0.5
        nxt[hit_high, move] = high_edge + 0.5
        if hazard is not None:
            step_clear = _segment_clearance(
                pos, nxt, np.asarray(hazard["center"], dtype=np.float64), r, float(hazard["radius"])
            )
            newly = (step_clear <= 0) & (safe_steps == BLOCK)
            safe_steps[newly] = k
            clear = np.minimum(clear, step_clear)
        pos = nxt
        path[:, k + 1] = pos
    return clear, safe_steps, path


def pusht_rollout(env, start_pos, start_vel, actions, hazard):
    """Pusher-only PD dynamics (contacts ignored) for N blocks. actions: (N, 5, 2) raw."""
    u = env.unwrapped
    dt, kp, kv, scale = float(u.dt), float(u.k_p), float(u.k_v), float(u.action_scale)
    substeps = int(1 / (dt * u.control_hz))
    n = actions.shape[0]
    pos = np.repeat(np.asarray(start_pos, dtype=np.float64)[None, :2], n, axis=0)
    vel = np.repeat(np.asarray(start_vel, dtype=np.float64)[None, :2], n, axis=0)
    clear = np.full(n, np.inf)
    safe_steps = np.full(n, BLOCK, dtype=np.int64)
    path = np.zeros((n, BLOCK + 1, 2))
    path[:, 0] = pos
    centre = None if hazard is None else np.asarray(hazard["center"], dtype=np.float64)
    for k in range(BLOCK):
        target = pos + actions[:, k, :].astype(np.float64) * scale if u.relative else actions[:, k, :]
        for _ in range(substeps):
            acc = kp * (target - pos) + kv * (0.0 - vel)
            vel = vel + acc * dt
            nxt = pos + vel * dt
            if hazard is not None:
                step_clear = _segment_clearance(pos, nxt, centre, 15.0, float(hazard["radius"]))
                newly = (step_clear <= 0) & (safe_steps == BLOCK)
                safe_steps[newly] = k
                clear = np.minimum(clear, step_clear)
            pos = nxt
        path[:, k + 1] = pos
    return clear, safe_steps, path


REACHER_LINK = 0.12  # both reacher links, as in geometry.recorded
REACHER_TIP = 0.01  # fingertip radius (checked by PhysicalReader.validate)
REACHER_REPEAT = 2  # DMControlWrapper action repeat; control timestep == physics timestep for reacher


def reacher_rollout(env, actions, hazard):
    """Exact physics oracle for the Reacher planner: roll the N torque blocks forward with mujoco.rollout
    from the live physics state (action repeat 2, constant control per planner action) and read the
    fingertip from the rolled-out joint angles. Returns clearance, safe steps and tip path (N, 6, 2)."""
    import mujoco
    import mujoco.rollout as mjro

    physics = env.unwrapped.env.physics
    model = physics.model.ptr
    full = mujoco.mjtState.mjSTATE_FULLPHYSICS
    cache = CONTEXT.setdefault("reacher_cache", {})
    if cache.get("model") is not model:
        cache.clear()
        cache.update(model=model, scratch=[mujoco.MjData(model)], nstate=mujoco.mj_stateSize(model, full))
    state0 = np.empty(cache["nstate"])
    mujoco.mj_getState(model, physics.data.ptr, state0, full)
    n = actions.shape[0]
    control = np.repeat(actions.astype(np.float64), REACHER_REPEAT, axis=1)  # (N, 10, nu)
    state, _ = mjro.rollout(model, cache["scratch"], np.tile(state0, (n, 1)), control)
    q = np.asarray(state)[:, :, 1:3]  # qpos[0:2]: shoulder, wrist (time is state[0])
    tip = REACHER_LINK * np.stack([np.cos(q[..., 0]) + np.cos(q[..., 0] + q[..., 1]),
                                   np.sin(q[..., 0]) + np.sin(q[..., 0] + q[..., 1])], axis=-1)  # (N, 10, 2)
    q0 = np.asarray(physics.data.qpos[:2])
    tip0 = REACHER_LINK * np.array([np.cos(q0[0]) + np.cos(q0[0] + q0[1]), np.sin(q0[0]) + np.sin(q0[0] + q0[1])])
    path = np.concatenate([np.repeat(tip0[None, None], n, axis=0), tip[:, REACHER_REPEAT - 1 :: REACHER_REPEAT]], axis=1)  # (N, 6, 2)
    if hazard is None:
        return np.full(n, np.inf), np.full(n, BLOCK, dtype=np.int64), path
    centre = np.asarray(hazard["center"][:2], dtype=np.float64)
    full_path = np.concatenate([np.repeat(tip0[None, None], n, axis=0), tip], axis=1)  # (N, 11, 2)
    seg = np.stack([_segment_clearance(full_path[:, j], full_path[:, j + 1], centre, REACHER_TIP, float(hazard["radius"]))
                    for j in range(full_path.shape[1] - 1)], axis=1)  # (N, 10)
    per_action = seg.reshape(n, BLOCK, REACHER_REPEAT).min(axis=2)
    clear = per_action.min(axis=1)
    unsafe = per_action <= 0
    safe_steps = np.where(unsafe.any(1), unsafe.argmax(1), BLOCK).astype(np.int64)
    return clear, safe_steps, path


def rollout(raw_actions):
    """Oracle for the current decision context. raw_actions: (N, 5, action_dim)."""
    task, env, hazard = CONTEXT["task"], CONTEXT["env"], CONTEXT["hazard"]
    if task == "reacher":
        return reacher_rollout(env, raw_actions, hazard)
    if task == "tworoom":
        start = np.asarray(env.unwrapped.agent_position.numpy(), dtype=np.float64)
        return tworoom_rollout(env, start, raw_actions, hazard)
    if task == "pusht":
        u = env.unwrapped
        clear, safe_steps, path = pusht_rollout(
            env, np.asarray(u.agent.position), np.asarray(u.agent.velocity), raw_actions, hazard
        )
        CONTEXT["last_path"] = path
        return clear, safe_steps, path
    raise ValueError("No kinematic oracle for task " + task)


def polygon_distances(points, polygon):
    """Vectorised geometry.polygon_distance: (M, 2) points -> (M,) signed distances, negative inside."""
    pts = np.asarray(points, dtype=np.float64)
    poly = np.asarray(polygon, dtype=np.float64)
    edges = np.roll(poly, -1, axis=0) - poly  # (E, 2)
    rel = pts[:, None, :] - poly[None, :, :]  # (M, E, 2)
    cross = edges[None, :, 0] * rel[:, :, 1] - edges[None, :, 1] * rel[:, :, 0]
    inside = np.all(cross >= -1e-12, axis=1) | np.all(cross <= 1e-12, axis=1)
    denom = np.maximum((edges ** 2).sum(1), 1e-30)
    amount = np.clip((rel * edges[None]).sum(2) / denom[None], 0, 1)
    closest = rel - amount[:, :, None] * edges[None]
    dist = np.sqrt((closest ** 2).sum(2)).min(1)
    return np.where(inside, -dist, dist)


def _pusht_decision_geometry():
    """Decision-time geometry for the 13-feature scorer, cached per decision step."""
    key = ("geometry", CONTEXT["step"])
    if CONTEXT.get(key) is None:
        u = CONTEXT["env"].unwrapped
        hazard = CONTEXT["hazard"]
        polys = [np.array([u.block.local_to_world(v) for v in shape.get_vertices()]) for shape in u.block.shapes]
        centre = np.asarray(u.block.position, dtype=np.float64)
        hz = np.asarray(hazard["center"][:2], dtype=np.float64)
        g0 = min(polygon_distance(hz, poly) - float(hazard["radius"]) for poly in polys)
        CONTEXT[key] = dict(
            polys=polys, centre=centre, hazard=hz,
            g0=float(np.clip(g0 / 100.0, -1, 1)),
            hrel=np.clip((hz - centre) / 512.0, -1, 1),
            prel=np.clip((np.asarray(u.agent.position, dtype=np.float64) - centre) / 512.0, -1, 1),
        )
    return CONTEXT[key]


def learned_key(values, safe_steps, step, paths=None):
    """Bounded network key; lower is better for topk.

    Six dimensionless features (v4 vocabulary) when the genotype mask has six entries; thirteen when it
    has thirteen (PushT only): the six plus the T's clearance to the hazard at the decision, the
    candidate's minimum pusher-to-T distance along its predicted pusher path, the pusher's predicted end
    position relative to the T centre, and the hazard centre and pusher start relative to the T centre.
    These are the features v4relabel_eval evolves on, computed here from the pusher-only rollout.
    """
    g = CONTEXT["genotype"]
    mask = np.asarray(g["mask"], dtype=np.float64)
    n_features = len(mask)
    v = np.asarray(values, dtype=np.float64)
    k = len(v)
    rank = np.argsort(np.argsort(v)) / max(k - 1, 1)
    z = (v - v.mean()) / (v.std() + 1e-9)
    cols = [
        safe_steps.astype(np.float64) / BLOCK,
        rank,
        np.clip(z, -5, 5) / 5,
        np.full(k, float(np.mean(safe_steps == BLOCK))),
        np.zeros(k),
        np.full(k, min(step / 200.0, 1.0) if n_features == 6 else step / 200.0),
    ]
    if n_features == 13:
        geo = _pusht_decision_geometry()
        if paths is None or len(paths) != k:
            raise ValueError("13-feature key needs one predicted pusher path per candidate")
        pts = np.asarray(paths)[:, 1:, :]  # positions after each of the five actions, (k, 5, 2)
        pb = np.full(k, np.inf)
        flat = pts.reshape(-1, 2)
        for poly in geo["polys"]:
            pb = np.minimum(pb, polygon_distances(flat, poly).reshape(k, -1).min(axis=1) - 15.0)
        end = np.clip((np.asarray(paths)[:, -1, :] - geo["centre"]) / 512.0, -1, 1)
        # offline layout: m, r, z, f, g0, pb, ex, ey, b, hx, hy, px, py
        cols = cols[:4] + [np.full(k, geo["g0"]), np.clip(pb / 50.0, -1, 1), end[:, 0], end[:, 1], cols[5],
                           np.full(k, geo["hrel"][0]), np.full(k, geo["hrel"][1]), np.full(k, geo["prel"][0]), np.full(k, geo["prel"][1])]
    x = np.stack(cols, axis=-1) * mask[None, :]
    w = np.asarray(g["params"], dtype=np.float64)
    if g["arch"] == "linear":
        net = x @ w[:n_features]
    else:
        h = g["hidden"]
        hidden = np.tanh(x @ w[: n_features * h].reshape(n_features, h))
        net = hidden @ w[n_features * h : (n_features + 1) * h]
    if g.get("form") == "veto":
        # learned feasibility predicate, native cost within the predicted-safe set (the hand rule's form,
        # with the network's verdict in place of the oracle's): candidates scored above the threshold
        # form the preferred tier and the native z-score orders inside each tier
        tier = (np.tanh(net) > float(g.get("threshold", 0.0))).astype(np.float64)
        budget = g.get("budget")
        if budget is not None and tier.any() and not tier.all():
            # budgeted veto (the dial's third knob): apply the learned tier only when the best
            # predicted-safe candidate costs at most `budget` x the population's cost spread more
            # than the native best; otherwise this decision is left to the native planner
            safe = tier > 0
            if z[safe].min() > z.min() + float(budget) * (z.max() - z.min()):
                return z.copy()
        return -tier * 1e3 + z
    return -np.tanh(net) * 1e3 + z  # lower = preferred


def tiers_for(raw_actions):
    clear, safe_steps, _ = rollout(raw_actions)
    rule = CONTEXT["rule"]
    tiers = safe_steps.astype(np.float64)
    if rule == "shuffle" and CONTEXT["hazard"] is not None:
        tiers = CONTEXT["rng"].permutation(tiers)
    return clear, safe_steps, tiers


# ---------------------------------------------------------------- patched CEM


def _raw(native, normalized, raw_bounds):
    flat = normalized.reshape(-1, native.action_dim)
    raw = native.scaler.inverse_transform(flat).astype(np.float32)
    raw = np.clip(raw, raw_bounds[0], raw_bounds[1])
    return raw.reshape(normalized.shape[0], BLOCK, native.action_dim)


def _nudge_reference(raw_block):
    """The 'do less' action: zero displacement (TwoRoom, relative PushT), zero torque (Reacher), or the
    pusher's current position when PushT actions are absolute targets."""
    ref = np.zeros_like(raw_block, dtype=np.float64)
    if CONTEXT["task"] == "pusht" and not CONTEXT["env"].unwrapped.relative:
        ref[:] = np.asarray(CONTEXT["env"].unwrapped.agent.position, dtype=np.float64)[None, :2]
    return ref


def _nudge(native, raw_block, base_clear, base_steps, margin, raw_bounds):
    """Gain ladder on the executed block (5, d): first gain whose block the oracle clears with the margin;
    otherwise the most clear steps, then the largest clearance, applied only if it beats gain 1."""
    ref = _nudge_reference(raw_block)
    gains = np.asarray(GAINS, dtype=np.float64)
    cands = ref[None] + gains[:, None, None] * (raw_block.astype(np.float64)[None] - ref[None])
    cands = np.clip(cands, raw_bounds[0], raw_bounds[1]).astype(np.float32)
    clear, steps, _ = rollout(cands)
    ok = (steps == BLOCK) & (clear >= margin)
    if ok.any():
        j = int(np.argmax(ok))
    else:
        j = int(np.lexsort((-gains, -clear, -steps))[0])
    better = bool(ok[j]) or steps[j] > base_steps or (steps[j] == base_steps and clear[j] > base_clear)
    out = dict(gain=float(gains[j]) if better else 1.0, base_clear=base_clear, base_steps=base_steps,
               clear=float(clear[j]), steps=int(steps[j]), cleared=bool(ok[j]))
    if better:
        flat = cands[j].reshape(-1, native.action_dim)
        out["normalized"] = native.scaler.transform(flat).astype(np.float32).reshape(BLOCK, native.action_dim)
    return out


def patched_cem(native, rgb, current, target, generator, normalized_prior, normalized_bounds, raw_bounds):
    active = CONTEXT.get("active", False)
    rule = CONTEXT.get("rule") if active else None
    hooked = active and rule in ("sort", "shuffle", "learned", "combined") and CONTEXT.get("hazard") is not None
    prior = torch.as_tensor(normalized_prior, dtype=torch.float32, device=native.device).reshape(1, 1, BLOCK * native.action_dim).clone()
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

    first_noise = None
    last_population = None
    first_population = None
    best_elite = None
    pop_paths = None
    for iteration in range(ITERATIONS):
        noise = torch.randn(1, SAMPLES, 1, BLOCK * native.action_dim, generator=generator, device=native.device)
        if iteration == 0:
            first_noise = noise[0, 1, 0].cpu().tolist()
            if getattr(native, "capture_initial_noise", False):
                native.initial_standard_noise = noise.detach().cpu().numpy().copy()
        candidates = noise * std[:, None] + mean[:, None]
        candidates[:, 0] = mean
        values, bounded = cost(candidates)
        if hooked or (active and iteration in (0, ITERATIONS - 1)):
            pop_norm = bounded[0, :, 0, :].detach().cpu().numpy()
            raw = _raw(native, pop_norm, raw_bounds)
            if CONTEXT.get("hazard") is not None:
                clear, safe_steps, tiers = tiers_for(raw)
                pop_paths = CONTEXT.get("last_path")
            else:
                clear = np.full(SAMPLES, np.inf)
                safe_steps = np.full(SAMPLES, BLOCK, dtype=np.int64)
                tiers = safe_steps.astype(np.float64)
                pop_paths = None
        if hooked:
            if rule == "learned":
                pop_values_np = values[0].detach().cpu().numpy()
                final_only = CONTEXT["genotype"].get("scope") == "final"
                if final_only and iteration < ITERATIONS - 1:
                    keyed = values  # native elite selection on early iterations; the key acts at the last one
                else:
                    keyed = torch.as_tensor(
                        learned_key(pop_values_np, safe_steps, CONTEXT["step"], paths=pop_paths),
                        dtype=torch.float32, device=native.device,
                    )[None, :]
            else:
                penalty = torch.as_tensor((BLOCK - tiers) * BIG, dtype=torch.float32, device=native.device)[None, :]
                keyed = values + penalty
            elite = torch.topk(keyed, ELITES, dim=1, largest=False).indices
            best = int(torch.argmin(keyed[0]).item())
            best_elite = dict(
                index=best,
                normalized=bounded[0, best, 0, :].detach().cpu().numpy().copy(),
                cost=float(values[0, best].item()),
                tier=float(tiers[best]),
                clear=float(clear[best]),
                safe_steps=int(safe_steps[best]),
            )
        else:
            elite = torch.topk(values, ELITES, dim=1, largest=False).indices
        if active and iteration == 0:
            first_population = dict(
                normalized=pop_norm.astype(np.float16),
                values=values[0].detach().cpu().numpy().astype(np.float32),
                clear=clear.astype(np.float32),
                safe_steps=safe_steps.astype(np.int8),
            )
        if active and iteration == ITERATIONS - 1:
            last_population = dict(
                normalized=pop_norm.astype(np.float16),
                values=values[0].detach().cpu().numpy().astype(np.float32),
                clear=clear.astype(np.float32),
                safe_steps=safe_steps.astype(np.int8),
            )
        elites = candidates[torch.arange(1, device=native.device)[:, None], elite]
        mean = elites.mean(dim=1)
        std = elites.std(dim=1).clamp_min(STD_FLOOR)
    values, effective = cost(torch.stack((prior, mean), dim=1))
    improved = bool((values[0, 1] < values[0, 0]).item())
    chosen = effective[:, 1] if improved else effective[:, 0]
    executed_kind = "mean" if improved else "prior"
    assert work_calls == 31 and work_transitions == 9002
    normalized = chosen[0, 0].cpu().numpy().reshape(BLOCK, native.action_dim)
    hook_detail = {}
    nudged = False
    if active:
        pair_norm = effective[0, :, 0, :].detach().cpu().numpy()
        pair_raw = _raw(native, pair_norm, raw_bounds)
        pair_paths = None
        if CONTEXT.get("hazard") is not None:
            pair_clear, pair_steps, _ = rollout(pair_raw)
            pair_paths = CONTEXT.get("last_path")
        else:
            pair_clear, pair_steps = np.full(2, np.inf), np.full(2, BLOCK)
        native_index = 1 if improved else 0
        hook_detail = dict(
            prior_clear=float(pair_clear[0]), prior_safe_steps=int(pair_steps[0]),
            mean_clear=float(pair_clear[1]), mean_safe_steps=int(pair_steps[1]),
            native_choice=executed_kind, native_choice_safe_steps=int(pair_steps[native_index]),
            best_elite=None if best_elite is None else {k: v for k, v in best_elite.items() if k != "normalized"},
        )
        if hooked:
            options = [
                ("prior", float(pair_steps[0]), float(values[0, 0].item()), pair_norm[0]),
                ("mean", float(pair_steps[1]), float(values[0, 1].item()), pair_norm[1]),
                ("elite", float(best_elite["safe_steps"]), best_elite["cost"], best_elite["normalized"]),
            ]
            if rule == "shuffle":
                shuffled = CONTEXT["rng"].permutation([o[1] for o in options])
                options = [(o[0], float(shuffled[i]), o[2], o[3]) for i, o in enumerate(options)]
            native_steps = options[native_index][1]
            if native_steps < BLOCK:
                if rule == "learned":
                    # score the three options with the learned key against the final population statistics
                    opt_steps = np.array([o[1] for o in options])
                    opt_costs = np.array([o[2] for o in options])
                    pool_vals = np.concatenate([pop_values_np, opt_costs])
                    pool_steps = np.concatenate([safe_steps, opt_steps.astype(np.int64)])
                    pool_paths = None
                    if pop_paths is not None and pair_paths is not None:
                        pool_paths = np.concatenate([pop_paths, pair_paths[:2], pop_paths[best_elite["index"]][None]], axis=0)
                    keys = learned_key(pool_vals, pool_steps, CONTEXT["step"], paths=pool_paths)[-3:]
                    pick = options[int(np.argmin(keys))]
                else:
                    pick = min(options, key=lambda o: (-o[1], o[2]))
                executed_kind = pick[0]
                normalized = np.asarray(pick[3], dtype=np.float32).reshape(BLOCK, native.action_dim)
        hook_detail["executed_kind"] = executed_kind
        raw_exec = _raw(native, normalized.reshape(1, -1), raw_bounds)
        if CONTEXT.get("hazard") is not None:
            exec_clear, exec_steps, exec_path = rollout(raw_exec)
        else:
            exec_clear, exec_steps, exec_path = np.array([np.inf]), np.array([BLOCK]), None
        nudge = None
        if rule in ("nudge", "combined") and CONTEXT.get("hazard") is not None:
            margin = NUDGE_MARGIN.get(CONTEXT["task"], 0.0)
            if exec_steps[0] < BLOCK or exec_clear[0] < margin:
                nudge = _nudge(native, raw_exec[0], float(exec_clear[0]), int(exec_steps[0]), margin, raw_bounds)
                if nudge["gain"] != 1.0:
                    normalized = nudge["normalized"]
                    raw_exec = _raw(native, normalized.reshape(1, -1), raw_bounds)
                    exec_clear, exec_steps, exec_path = rollout(raw_exec)
                    nudged = True
        hook_detail["nudge"] = None if nudge is None else {k: v for k, v in nudge.items() if k != "normalized"}
        hook_detail.update(executed_clear=float(exec_clear[0]), executed_safe_steps=int(exec_steps[0]))
        CONTEXT["record"] = dict(
            step=CONTEXT["step"],
            population=last_population,
            first_population=first_population,
            executed_normalized=normalized.astype(np.float32),
            executed_kind=executed_kind,
            executed_clear=float(exec_clear[0]),
            executed_safe_steps=int(exec_steps[0]),
            executed_path=None if exec_path is None else exec_path[0].astype(np.float32),
            prior_clear=float(pair_clear[0]),
            mean_clear=float(pair_clear[1]),
            native_choice=hook_detail["native_choice"],
            gate_open=nudge is not None,
            nudge_gain=1.0 if nudge is None else float(nudge["gain"]),
        )
    raw = native.scaler.inverse_transform(normalized).astype(np.float32)
    raw = np.clip(raw, raw_bounds[0], raw_bounds[1]).astype(np.float32)
    roundtrip_error = float(np.max(np.abs(native.scaler.transform(raw) - normalized)))
    assert raw.shape == (BLOCK, native.action_dim) and np.isfinite(raw).all()
    # TargetGaussian.choose replaces the returned block with the raw prior whenever this flag is
    # False, so a hooked pick other than the prior must set it True; the native path keeps `improved`.
    keep_flag = improved if not hooked else (executed_kind != "prior")
    if nudged:
        keep_flag = True  # the nudged block is not the raw prior
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
    detail["v4"] = hook_detail
    return raw, detail


def install():
    """Replace the planner's CEM binding; the original stays importable."""
    from anchored_planning import planners

    if getattr(planners, "_v4_installed", False):
        return
    planners._native_cem = planners.cem
    planners.cem = patched_cem
    planners._v4_installed = True
