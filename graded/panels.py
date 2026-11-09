"""Graded study panels (9 Nov 2026): calibration natives, frozen step, level 1-4 placement, panel files.

Rules: hazard_design/DESIGN.md (locked). Format and sizes: graded/INTERFACES.md.

The frozen planner never sees the hazard, so one clean, recorded native run per group serves all four levels.

AP tasks (tworoom, pusht, reacher) talk to residual.worker processes through a job directory:
  python -m graded.panels ap <task> --queue Q --out OUT [--panels graded/panels] [--runs RUNS]
    1 cases     clean cases of the training-type splits and the held-out split
    2 natives   one clean native recorded episode per candidate group  (job ids cal-<task>-nat-<group>)
    3 build     admission (native completes, placement feasible), frozen step = median native travel per
                decision over the groups used, levels 1-4, the panel cases
    4 checks    determinism (2 clean reruns, no recording: parity must match) and hazard blindness
                (2 level 3 runs: parity must match the clean run; monitor clearance vs the drawn prediction)
Basketball (TD-MPC2) runs natives in-process (same seeding and eval mode as residual_tdmpc2 / td_levels):
  python -m graded.panels basketball --device N --out OUT [--panels graded/panels]
"""

import argparse
import fcntl
import glob
import gzip
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

BODY = {"tworoom": 7.0, "pusht": 15.0, "reacher": 0.01}
SUCCESS = {"tworoom": 16.0, "pusht": 20.0, "reacher": 0.018}  # success region around the body's goal
LIMIT = {"tworoom": 224.0, "pusht": 512.0, "reacher": 0.24}
BLOCK = 5
TRAIN_SPLITS = ("architecture_train", "pool_train", "calibration", "architecture_validation", "gate")
TEST_SPLITS = ("deployment",)
N_TRAIN, N_SELECT, N_TEST = 6, 2, 6
GRAZE = 0.25


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    tmp.replace(path)


def merge_summary(panels_dir, env, entry):
    panels_dir = Path(panels_dir)
    panels_dir.mkdir(parents=True, exist_ok=True)
    with open(panels_dir / ".summary.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = panels_dir / "summary.json"
        data = json.loads(path.read_text()) if path.exists() else {}
        data[env] = entry
        write_json(path, data)


# ------------------------------------------------------------------ planar geometry (AP)


def travel_hops(path):
    path = np.asarray(path, float)[:, :2]
    hops = np.linalg.norm(path[BLOCK::BLOCK] - path[:-BLOCK:BLOCK], axis=1)
    return hops[hops > 1e-9]


def clearance(path, center, radius, body):
    return float(np.min(np.linalg.norm(path[:, :2] - np.asarray(center, float)[:2], axis=1)) - radius - body)


def blocked(task, frame, center, radius):
    """Fraction of the contact disc on walls (dark pixels) or outside the arena (AP 2-D tasks)."""
    if task == "reacher":
        reach = np.linalg.norm(np.asarray(center, float)[:2])
        return float(reach > LIMIT["reacher"] - radius)
    h, w = frame.shape[:2]
    k = w / LIMIT[task]
    c = np.asarray(center, float)[:2] * k
    rb = (radius + BODY[task]) * k
    yy, xx = np.mgrid[0:h, 0:w]
    inside = (xx - c[0]) ** 2 + (yy - c[1]) ** 2 <= rb**2
    area = np.pi * rb**2
    dark = frame.mean(axis=2) < 60
    return float(max(0.0, 1 - inside.sum() / area) + (dark & inside).sum() / max(1, inside.sum()))


def anchor_index(path, step, radius, body):
    """propose.py's anchor: halfway by distance, nudged until the contact ring is 2 steps from both ends.
    Fallback (recorded as 'relaxed'), only for paths that loop back near their start (TwoRoom doors): scan
    fractions 0.35-0.65 nearest 0.5 first with the ring 1.5 steps from both ends, which still keeps the ring
    outside every success region (TwoRoom 16 px = 1.3 steps)."""
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    arc = np.concatenate([[0], np.cumsum(seg)])

    def far(i, need):
        return np.linalg.norm(path[i] - path[0]) >= need and np.linalg.norm(path[i] - path[-1]) >= need

    need = radius + body + 2 * step
    for f in (0.5, 0.45, 0.55, 0.4, 0.6, 0.35, 0.65):
        i = int(np.searchsorted(arc, f * arc[-1]))
        i = min(max(i, 1), len(path) - 2)
        if far(i, need):
            return i, float(arc[i] / arc[-1]), "strict"
    need = radius + body + 1.5 * step
    fr = arc / arc[-1]
    for i in np.argsort(np.abs(fr - 0.5)):
        if 0.35 <= fr[i] <= 0.65 and 1 <= i <= len(path) - 2 and far(i, need):
            return int(i), float(fr[i]), "relaxed"
    raise ValueError("path too short for a mid-path hazard")


def place_levels(task, path, step, side, frame, goal):
    """Levels 1-4 on one clean native path. side = +1 (left normal) or -1, used by levels 1 and 2."""
    path = np.asarray(path, float)[:, :2]
    body, radius = BODY[task], step
    anchor, frac, rule = anchor_index(path, step, radius, body)
    lo, hi = max(0, anchor - BLOCK), min(len(path) - 1, anchor + BLOCK)
    tangent = path[hi] - path[lo]
    tangent /= np.linalg.norm(tangent)
    normal = np.array([-tangent[1], tangent[0]])
    a = path[anchor]

    def at_clearance(target, s):
        lo_, hi_ = 0.0, 10 * (radius + body + 4 * step)
        for _ in range(80):
            m = 0.5 * (lo_ + hi_)
            if clearance(path, a + s * m * normal, radius, body) < target:
                lo_ = m
            else:
                hi_ = m
        return a + s * hi_ * normal

    # Level 1: nearest open spot (to the anchor) whose clearance from the whole path is 2-3 steps, on `side`.
    if task == "reacher":
        xs = np.linspace(-LIMIT[task], LIMIT[task], 121)
    else:
        xs = np.linspace(0, LIMIT[task], 161)
    gx, gy = np.meshgrid(xs, xs, indexing="ij")
    grid = np.stack([gx.ravel(), gy.ravel()], 1)
    d = np.sqrt(((grid[:, None, :] - path[None, :, :]) ** 2).sum(-1)).min(1) - radius - body
    ok = (d >= 2 * step) & (d <= 3 * step)
    if task == "reacher":
        rr = np.linalg.norm(grid, axis=1)
        ok &= (rr + radius <= LIMIT[task]) & (rr >= radius + 0.03)
    else:
        ok &= (grid.min(1) >= radius + body) & (grid.max(1) <= LIMIT[task] - radius - body)
    sgn = (grid - a) @ normal
    level1, l1_side = None, side
    for want in (side, -side):
        cand = np.flatnonzero(ok & (sgn * want > 0))
        cand = cand[np.argsort(np.linalg.norm(grid[cand] - a, axis=1))]
        for i in cand:
            if task != "reacher" and blocked(task, frame, grid[i], radius) > 0.02:
                continue
            level1, l1_side = grid[i], want
            break
        if level1 is not None:
            break
    if level1 is None:
        raise ValueError("no open off-path spot")

    l2 = at_clearance(-GRAZE * radius, side)
    r4 = float(max(radius, SUCCESS[task] + 0.25 * step - body))
    goal = np.asarray(goal, float)[:2]
    levels = [
        (1, l1_side, level1, radius),
        (2, side, l2, radius),
        (3, 0, a.copy(), radius),
        (4, 0, goal, r4),
    ]
    out = []
    for level, s, c, r in levels:
        cl = clearance(path, c, r, body)
        out.append({
            "level": level, "side": int(s), "center": [float(c[0]), float(c[1]), 0.0], "radius": float(r),
            "native_clearance": cl, "blocked": blocked(task, frame, c, r) if frame is not None else None,
        })
    return out, {"anchor_index": int(anchor), "anchor_arc_fraction": frac, "anchor_rule": rule}


def side_name(level, side):
    if level == 3:
        return "on"
    if level == 4:
        return "goal"
    return "left" if side == 1 else "right"


# ------------------------------------------------------------------ AP driver


def find_log(runs, task, job_id):
    hits = glob.glob(os.path.join(runs, "*", "artifacts", task, "episodes", job_id + ".jsonl.gz"))
    if not hits:
        raise FileNotFoundError(f"episode log for {job_id} not found under {runs}")
    with gzip.open(hits[0], "rt") as f:
        return json.loads(f.readline())


def goal_point(task, true_goal):
    from qd_adapters.geometry import recorded

    return np.asarray(recorded(task, np.asarray(true_goal, float))["reference"], float)[:2]


def run_ap(args):
    from residual.search import Queue

    task = args.task
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    q = Queue(args.queue)
    tag = args.tag
    test_only = task == "reacher"
    train_splits = () if test_only else TRAIN_SPLITS
    splits = list(train_splits) + list(TEST_SPLITS)

    # 1 cases
    jid = f"{tag}-{task}-cases-" + hashlib.sha256(",".join(splits).encode()).hexdigest()[:8]
    cases_path = out / f"cases-{jid}.json"
    if cases_path.exists():
        cases = json.loads(cases_path.read_text())
    else:
        q.submit({"job_id": jid, "kind": "cases", "splits": splits})
        cases = q.wait([jid])[0]["cases"]
        cases = [c for c in cases if c["condition"] == "clean"]
        write_json(cases_path, cases)
    print(json.dumps({"task": task, "clean_cases": len(cases)}), flush=True)

    # 2 natives (clean, recorded)
    media = out / "media"
    nat_ids = {}
    for c in cases:
        j = f"{tag}-{task}-nat-{c['source_group']}"
        nat_ids[c["id"]] = j
        if not (Path(args.queue) / "results" / (j + ".json")).exists():
            q.submit({"job_id": j, "phase": "calib", "case": c, "member": {"id": "native", "kind": "native"},
                      "native_scale": 1.0, "record": True, "media_dir": str(media)})
    t0 = time.time()
    res = dict(zip(nat_ids, q.wait(list(nat_ids.values()))))
    print(json.dumps({"task": task, "natives": len(res), "minutes": round((time.time() - t0) / 60, 1)}), flush=True)

    # 3 build
    runs = {}
    for c in cases:
        r = res[c["id"]]
        row = {"case": c, "result": r, "ok": bool(r.get("completed")) and bool(r.get("success"))}
        if r.get("completed"):
            data = np.load(r["media"])
            row["path"] = np.asarray(data["positions"], float)[:, :2]
            row["frame"] = data["frames"][0] if len(data["frames"]) else None
            row["goal"] = goal_point(task, find_log(args.runs, task, nat_ids[c["id"]])["true_goal"])
        runs[c["id"]] = row
    admitted = [cid for cid, rw in runs.items() if rw["ok"]]
    step = float(np.median(np.concatenate([travel_hops(runs[c]["path"]) for c in admitted])))
    # Select groups in split order; then freeze the step over the groups used and re-place (twice to settle).
    for _ in range(3):
        chosen = {"train": [], "select": [], "test": []}
        placed, reasons = {}, {}
        for cid in [c["id"] for c in cases]:
            rw = runs[cid]
            split = rw["case"]["row"]["split"]
            pool = "test" if split in TEST_SPLITS else "trainsel"
            if not rw["ok"]:
                reasons[cid] = "native did not complete" if rw["result"].get("completed") else "error"
                continue
            if pool == "test":
                if len(chosen["test"]) >= N_TEST:
                    continue
                target = "test"
            else:
                if len(chosen["train"]) < N_TRAIN:
                    target = "train"
                elif len(chosen["select"]) < N_SELECT:
                    target = "select"
                else:
                    continue
            k = len(chosen[target]) + (0 if target != "select" else N_TRAIN)
            side = 1 if k % 2 == 0 else -1
            try:
                placed[cid] = place_levels(task, rw["path"], step, side, rw["frame"], rw["goal"])
            except ValueError as e:
                reasons[cid] = str(e)
                continue
            if task != "reacher":
                # The PushT pusher can leave the rendered 512 x 512 arena; keep the on-path hazard inside it.
                a3 = next(lv for lv in placed[cid][0] if lv["level"] == 3)["center"]
                if not (0 <= a3[0] <= LIMIT[task] and 0 <= a3[1] <= LIMIT[task]):
                    reasons[cid] = "anchor outside the arena"
                    del placed[cid]
                    continue
            chosen[target].append(cid)
        used = chosen["train"] + chosen["select"] + chosen["test"]
        new_step = float(np.median(np.concatenate([travel_hops(runs[c]["path"]) for c in used])))
        if abs(new_step - step) < 1e-9:
            break
        step = new_step

    panel = []
    for split in ("train", "select", "test"):
        for cid in chosen[split]:
            rw = runs[cid]
            c = rw["case"]
            levels, meta = placed[cid]
            group = f"g{int(c['source_group']):04d}"
            for lv in levels:
                if lv["level"] == 4 and split != "test":
                    continue
                pid = f"{task}-{group}-L{lv['level']}-{side_name(lv['level'], lv['side'])}"
                hazard = {"center": lv["center"], "radius": lv["radius"]}
                ap_case = dict(c, id=pid, condition=f"L{lv['level']}", hazard=hazard)
                panel.append({
                    "id": pid, "env": task, "group": group, "level": lv["level"], "split": split,
                    "side": lv["side"], "step": step, "body_radius": BODY[task],
                    "hazard": hazard, "goal": [float(v) for v in rw["goal"]], "ap_case": ap_case,
                    "native": {
                        "success": bool(rw["result"]["success"]),
                        "violation": bool(lv["native_clearance"] < 0),
                        "min_clear_steps": float(lv["native_clearance"] / step),
                        "actions": int(rw["result"]["actions"]),
                        "parity_sha256": rw["result"]["parity_sha256"],
                        "clean_job": nat_ids[cid],
                    },
                    "placement": {"blocked": lv["blocked"], **meta,
                                  "goal_end_steps": float(np.linalg.norm(rw["path"][-1] - rw["goal"]) / step)},
                })

    # 4 checks: determinism (clean rerun, no recording) and hazard blindness (level 3, no recording)
    picks = ([chosen["train"][0]] if chosen["train"] else []) + ([chosen["test"][0]] if chosen["test"] else [])
    if test_only and len(chosen["test"]) > 1:
        picks = chosen["test"][:2]
    jobs = {}
    for cid in picks:
        c = runs[cid]["case"]
        g = c["source_group"]
        jobs[f"{tag}-{task}-det-{g}"] = ("det", cid, c)
        l3 = next(p for p in panel if p["ap_case"]["source_group"] == g and p["level"] == 3)
        hz = hashlib.sha256(json.dumps(l3["hazard"]).encode()).hexdigest()[:6]
        jobs[f"{tag}-{task}-chkL3-{g}-{hz}"] = ("L3", cid, l3["ap_case"])
    for j, (_, _, case) in jobs.items():
        if not (Path(args.queue) / "results" / (j + ".json")).exists():
            q.submit({"job_id": j, "phase": "calib", "case": case, "member": {"id": "native", "kind": "native"},
                      "native_scale": 1.0, "record": False})
    checks = []
    for j, r in zip(jobs, q.wait(list(jobs))):
        what, cid, case = jobs[j]
        clean = runs[cid]["result"]
        ent = {"job": j, "check": what, "case": case["id"], "completed": r.get("completed"),
               "parity_equal": r.get("parity_sha256") == clean["parity_sha256"],
               "actions": r.get("actions"), "clean_actions": clean["actions"], "error": r.get("error")}
        if what == "L3":
            p = next(p for p in panel if p["id"] == case["id"])
            ent.update(monitor_violation=r.get("violation"), monitor_swept_min_steps=(
                r["swept_min_clearance"] / step if r.get("swept_min_clearance") is not None else None),
                predicted_min_clear_steps=p["native"]["min_clear_steps"])
        checks.append(ent)
    deterministic = all(c["completed"] and c["parity_equal"] for c in checks)

    panels_dir = Path(args.panels)
    want = {"train": 0 if test_only else 3 * N_TRAIN, "select": 0 if test_only else 3 * N_SELECT, "test": 4 * N_TEST}
    full = {s: sum(p["split"] == s for p in panel) for s in ("train", "select", "test")} == want
    write_json(panels_dir / (f"{task}.json" if full else f"{task}.short.json"), panel)
    by_level = {}
    for p in panel:
        k = f"{p['split']}-L{p['level']}"
        e = by_level.setdefault(k, {"cases": 0, "native_success": 0, "native_violation": 0, "min_clear_steps": []})
        e["cases"] += 1
        e["native_success"] += p["native"]["success"]
        e["native_violation"] += p["native"]["violation"]
        e["min_clear_steps"].append(round(p["native"]["min_clear_steps"], 3))
    split_of = {c["id"]: c["row"]["split"] for c in cases}
    entry = {
        "env": task, "step": step, "body_radius": BODY[task], "hazard_radius_L1_3": step,
        "radius_L4": float(max(step, SUCCESS[task] + 0.25 * step - BODY[task])),
        "candidates": {s: sum(split_of[c] == s for c in split_of) for s in splits},
        "admitted": {s: sum(split_of[c] == s and runs[c]["ok"] for c in split_of) for s in splits},
        "groups": {k: [runs[c]["case"]["source_group"] for c in v] for k, v in chosen.items()},
        "group_splits": {k: [split_of[c] for c in v] for k, v in chosen.items()},
        "rejected": {runs[c]["case"]["source_group"]: why for c, why in reasons.items()},
        "panel_sizes": {s: sum(p["split"] == s for p in panel) for s in ("train", "select", "test")},
        "native_by_level": by_level, "checks": checks, "deterministic_and_hazard_blind": deterministic,
        "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    merge_summary(panels_dir, task, entry)
    write_json(out / "summary.json", entry)
    if not full:
        (panels_dir.parent / f"PANELS_SHORT_{task}").write_text(json.dumps(entry["panel_sizes"]) + "\n")
        print(json.dumps({"short": entry["panel_sizes"], "want": want}), flush=True)
        return
    (panels_dir.parent / f"PANELS_READY_{task}").write_text(json.dumps(
        {"panel": str(panels_dir / f"{task}.json"), "step": step, "sizes": entry["panel_sizes"],
         "deterministic_and_hazard_blind": deterministic}) + "\n")
    print(json.dumps({k: v for k, v in entry.items() if k != "native_by_level"}, indent=1), flush=True)


# ------------------------------------------------------------------ basketball (TD-MPC2)

TD_TRAIN_SEEDS = list(range(2000, 2032))  # fresh: the residual study used 1000-1015 and 5000-5047
TD_TEST_SEEDS = list(range(6000, 6032))


def box_distance(p, center, half):
    q = np.abs(np.asarray(p, float) - center) - half
    return float(np.linalg.norm(np.maximum(q, 0)) + min(np.max(q), 0))


def path_box_distance(samples, center, half):
    q = np.abs(np.asarray(samples, float) - center) - half
    return np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(q.max(1), 0)


def td_levels(eef_dec, obj_dec, samples, success_at, target, step, side):
    """Cube levels per DESIGN.md / td_box_levels.py on one traced native run (gripper point, body radius 0)."""
    n = len(eef_dec)
    half = np.full(3, step)
    carry_end = success_at if success_at > 0 else n - 1
    held = np.linalg.norm(eef_dec - obj_dec, axis=1) <= 0.03
    carry_start = carry_end
    while carry_start > 0 and held[carry_start - 1]:
        carry_start -= 1
    if carry_end - carry_start < 3:
        raise ValueError("no carry before success")
    seg = eef_dec[carry_start: carry_end + 1]
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(seg, axis=0), axis=1))]
    i = carry_start + int(np.searchsorted(arc, arc[-1] / 2))
    anchor = eef_dec[i]
    tangent = eef_dec[min(i + 2, n - 1)] - eef_dec[max(i - 2, 0)]
    tangent /= np.linalg.norm(tangent)
    lateral = np.cross(tangent, [0, 0, 1.0])
    if np.linalg.norm(lateral) < 1e-6:
        lateral = np.array([1.0, 0, 0])
    lateral /= np.linalg.norm(lateral)

    def offset_to(target_d, direction):
        a, b = 0.0, 0.5
        for _ in range(80):
            m = 0.5 * (a + b)
            if path_box_distance(samples, anchor + m * direction, half).min() < target_d:
                a = m
            else:
                b = m
        return anchor + b * direction

    g = np.linspace(-0.25, 0.25, 26)
    best = None
    for dx in g:
        for dy in g:
            for dz in np.linspace(-0.1, 0.2, 16):
                c = anchor + np.array([dx, dy, dz])
                if c[2] - half[2] < 0.01:
                    continue
                if np.dot(c - anchor, lateral) * side <= 0:
                    continue
                d = path_box_distance(samples, c, half).min()
                if 2 * step <= d <= 3 * step:
                    r = np.linalg.norm(c - anchor)
                    if best is None or r < best[0]:
                        best = (r, c)
    if best is None:
        raise ValueError("no off-path spot")
    offset = float(np.max(np.linalg.norm(eef_dec[carry_start: carry_end + 1] - obj_dec[carry_start: carry_end + 1], axis=1)))
    goal_half = np.full(3, 0.08 + offset + 0.25 * step)
    levels = [(1, side, best[1], half), (2, side, offset_to(-GRAZE * step, side * lateral), half),
              (3, 0, anchor, half), (4, 0, np.asarray(target, float), goal_half)]
    out = []
    for level, s, c, h in levels:
        d = path_box_distance(samples, c, h)
        out.append({"level": level, "side": int(s), "center": np.asarray(c).tolist(), "half": np.asarray(h).tolist(),
                    "native_min_distance_m": float(d.min())})
    meta = {"carry_decisions": [int(carry_start), int(carry_end)], "anchor_decision": int(i),
            "gripper_ball_offset_max_m": offset}
    return out, meta


def run_basketball(args):
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ["PYOPENGL_PLATFORM"] = os.environ["MUJOCO_GL"]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    import torch

    torch.backends.cudnn.benchmark = False
    from residual_tdmpc2 import systems

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    agent, _ = systems.load_agent("mw-basketball")
    s = systems.make_system("mw-basketball", hazard=None, render=False)

    def native(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        obs = s.reset(seed)
        h = hashlib.sha256()
        t_start = time.monotonic()
        for t in range(s.decisions):
            a = agent.act(torch.from_numpy(obs), t0=(t == 0), eval_mode=True).numpy()
            h.update(np.ascontiguousarray(a, dtype=np.float32).tobytes())
            obs, _, info = s.step(a)
        o = s.outcome(0.0)
        samples = [dict(x) for x in s.samples]
        return {"seed": seed, "success": o["success"], "target": o["target"], "parity_sha256": h.hexdigest(),
                "samples": samples, "seconds": time.monotonic() - t_start}

    cache = out / "natives.json"
    nat = json.loads(cache.read_text()) if cache.exists() else {}
    for seed in TD_TRAIN_SEEDS + TD_TEST_SEEDS:
        if str(seed) not in nat:
            nat[str(seed)] = native(seed)
            print(json.dumps({"seed": seed, "success": nat[str(seed)]["success"],
                              "s": round(nat[str(seed)]["seconds"], 1)}), flush=True)
            write_json(cache, nat)
    build_basketball(args, nat, native)


def build_basketball(args, nat, native):
    out = Path(args.out)

    def arrays(r):
        smp = np.array([x["eef"] for x in r["samples"]], float)
        obj = np.array([x["obj"] for x in r["samples"]], float)
        succ = [k for k, x in enumerate(r["samples"]) if x["success"] >= 1.0]
        success_at = (succ[0] + 1) // 2 if succ else -1  # sample 2k is the state after decision k
        return smp, smp[0::2], obj[0::2], success_at

    def hops(sd):
        # Travel per decision up to success (AP episodes stop at success; basketball runs on for 100 decisions).
        _, eef, _, success_at = arrays(nat[str(sd)])
        return np.linalg.norm(np.diff(eef[: success_at + 1], axis=0), axis=1)

    admitted = [sd for sd in TD_TRAIN_SEEDS + TD_TEST_SEEDS if nat[str(sd)]["success"]]
    step = float(np.median(np.concatenate([hops(sd) for sd in admitted])))
    for _ in range(3):
        chosen = {"train": [], "select": [], "test": []}
        placed, reasons = {}, {}
        for sd in TD_TRAIN_SEEDS + TD_TEST_SEEDS:
            r = nat[str(sd)]
            if not r["success"]:
                reasons[sd] = "native did not complete"
                continue
            if sd in TD_TEST_SEEDS:
                if len(chosen["test"]) >= N_TEST:
                    continue
                target = "test"
            elif len(chosen["train"]) < N_TRAIN:
                target = "train"
            elif len(chosen["select"]) < N_SELECT:
                target = "select"
            else:
                continue
            k = len(chosen[target]) + (0 if target != "select" else N_TRAIN)
            side = 1 if k % 2 == 0 else -1
            smp, eef, obj, success_at = arrays(r)
            try:
                placed[sd] = td_levels(eef, obj, smp, success_at, r["target"], step, side)
            except ValueError as e:
                reasons[sd] = str(e)
                continue
            chosen[target].append(sd)
        used = chosen["train"] + chosen["select"] + chosen["test"]
        new_step = float(np.median(np.concatenate([hops(sd) for sd in used])))
        if abs(new_step - step) < 1e-12:
            break
        step = new_step

    # determinism: two seeds rerun
    checks = []
    for sd in (chosen["train"][:1] + chosen["test"][:1]):
        again = native(sd)
        checks.append({"seed": sd, "parity_equal": again["parity_sha256"] == nat[str(sd)]["parity_sha256"],
                       "success": again["success"]})
    deterministic = all(c["parity_equal"] for c in checks)

    panel = []
    for split in ("train", "select", "test"):
        for sd in chosen[split]:
            levels, meta = placed[sd]
            r = nat[str(sd)]
            for lv in levels:
                if lv["level"] == 4 and split != "test":
                    continue
                c, h = np.asarray(lv["center"]), np.asarray(lv["half"])
                pid = f"basketball-s{sd}-L{lv['level']}-{side_name(lv['level'], lv['side'])}"
                panel.append({
                    "id": pid, "env": "basketball", "group": f"s{sd}", "level": lv["level"], "split": split,
                    "side": lv["side"], "step": step, "body_radius": 0.0,
                    "hazard": {"box_lo": (c - h).tolist(), "box_hi": (c + h).tolist(), "step_m": step},
                    "goal": [float(v) for v in r["target"]], "seed": sd,
                    "native": {"success": bool(r["success"]), "violation": bool(lv["native_min_distance_m"] <= 0),
                               "min_clear_steps": float(lv["native_min_distance_m"] / step),
                               "parity_sha256": r["parity_sha256"]},
                    "placement": meta,
                })
    panels_dir = Path(args.panels)
    write_json(panels_dir / "basketball.json", panel)
    by_level = {}
    for p in panel:
        k = f"{p['split']}-L{p['level']}"
        e = by_level.setdefault(k, {"cases": 0, "native_success": 0, "native_violation": 0, "min_clear_steps": []})
        e["cases"] += 1
        e["native_success"] += p["native"]["success"]
        e["native_violation"] += p["native"]["violation"]
        e["min_clear_steps"].append(round(p["native"]["min_clear_steps"], 3))
    entry = {
        "env": "basketball", "step": step, "body_radius": 0.0, "half_size_L1_3": step,
        "seed_ranges": {"train_select": [TD_TRAIN_SEEDS[0], TD_TRAIN_SEEDS[-1]],
                        "test": [TD_TEST_SEEDS[0], TD_TEST_SEEDS[-1]]},
        "admitted": {"train_select": sum(nat[str(s)]["success"] for s in TD_TRAIN_SEEDS),
                     "test": sum(nat[str(s)]["success"] for s in TD_TEST_SEEDS),
                     "of": len(TD_TRAIN_SEEDS)},
        "groups": chosen, "rejected": {str(k): v for k, v in reasons.items()},
        "panel_sizes": {s: sum(p["split"] == s for p in panel) for s in ("train", "select", "test")},
        "native_by_level": by_level, "checks": checks, "deterministic": deterministic,
        "native_eval_mode": True, "step_rule": "median gripper travel per decision up to success", "seeding": "torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); reset(seed)",
        "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    merge_summary(panels_dir, "basketball", entry)
    write_json(out / "summary.json", entry)
    (panels_dir.parent / "PANELS_READY_basketball").write_text(json.dumps(
        {"panel": str(panels_dir / "basketball.json"), "step": step, "sizes": entry["panel_sizes"],
         "deterministic": deterministic}) + "\n")
    print(json.dumps({k: v for k, v in entry.items() if k != "native_by_level"}, indent=1), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("ap")
    a.add_argument("task", choices=("tworoom", "pusht", "reacher"))
    a.add_argument("--queue", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--panels", default=str(Path(__file__).parent / "panels"))
    a.add_argument("--runs", default=os.path.join(os.environ.get("WS", "."), "runs/graded/calib-w"))
    a.add_argument("--tag", default="cal")
    b = sub.add_parser("basketball")
    b.add_argument("--device", default="7")
    b.add_argument("--out", required=True)
    b.add_argument("--panels", default=str(Path(__file__).parent / "panels"))
    args = ap.parse_args()
    if args.cmd == "ap":
        run_ap(args)
    else:
        run_basketball(args)


if __name__ == "__main__":
    main()
