"""Prospective source-group splits and hazard construction without policy outcomes."""

import hashlib

import numpy as np

from .geometry import SCALES, clearance, recorded

TASKS = ("tworoom", "pusht", "reacher", "cube")
CAPS = {"tworoom": 200, "pusht": 280, "reacher": 300, "cube": 150}
HORIZONS = {"tworoom": 100, "pusht": 140, "reacher": 150, "cube": 150}
SPLIT_COUNTS = {
    "gate": 4,
    "architecture_train": 4,
    "architecture_validation": 8,
    "pool_train": 4,
    "calibration": 16,
    "deployment": 16,
    "final_reserved": 128,
}


def stable_seed(*parts):
    text = ":".join(str(p) for p in parts).encode()
    return int.from_bytes(hashlib.sha256(text).digest()[:4], "big") % (2**31 - 1)


def split_groups(task, lengths, offsets, native_protocol, prior_episodes=()):
    """Read lengths/offsets only for the reserved final set; preserve AP memory."""
    forbidden = set(native_protocol["bank_train_episodes"][task]) | set(prior_episodes)
    forbidden.update(r["episode"] for r in native_protocol["studies"][task]["confirm"])
    # Reconstruct the completed checkpoint study's row sampling from metadata
    # only, including its recorded exclusion of the last valid row. No outcome
    # or observation is consulted. Additional known IDs may be supplied above.
    valid = np.concatenate(
        [
            np.arange(int(offset), int(offset) + max(0, int(length) - 25))
            for length, offset in zip(lengths, offsets)
        ]
    )
    for seed, count in ((0, 2), (42, 50), (43, 50), (44, 50)):
        chosen = valid[
            np.random.default_rng(seed).choice(len(valid) - 1, size=count, replace=False)
        ]
        forbidden.update(np.searchsorted(offsets, chosen, side="right") - 1)
    horizon = HORIZONS[task]
    legal = [i for i, length in enumerate(lengths) if i not in forbidden and length > horizon]
    legal.sort(key=lambda i: (stable_seed("split-v1", task, i), i))
    if len(legal) < sum(SPLIT_COUNTS.values()):
        raise ValueError("Insufficient disjoint source groups for the frozen split")
    result, cursor = {}, 0
    for split, count in SPLIT_COUNTS.items():
        result[split] = []
        for episode in legal[cursor : cursor + count]:
            start = int(offsets[episode])
            result[split].append(
                {
                    "episode": int(episode),
                    "global_start": start,
                    "global_goal": start + horizon,
                    "horizon": horizon,
                    "environment_seed": stable_seed("environment", task, episode),
                    "optimizer_seed": stable_seed("planner", task, episode),
                    "record_id": f"qd-v1-{task}-{split}-{episode}",
                    "split": split,
                }
            )
        cursor += count
    return result


def planned_gate_attempts(cases):
    """Engineering (two groups, two runs), seven policies per context, one witness per hazard."""
    return 4 + 7 * len(cases) + sum(case["condition"] != "clean" for case in cases)


def hazards(task, states):
    """Two near-demonstration hazards, chosen using recorded geometry only.

    Search order and distances are fixed globally. Accept the first geometric
    placement with positive clearance along the entire recorded path. Actual
    safe feasibility still requires a successful live replay in the gate.
    A side with no feasible placement is returned as None: an infeasible-panel
    outcome that is recorded, never replaced by another group or placement.
    """
    shapes = [recorded(task, state) for state in states]
    refs = np.array([s["reference"] for s in shapes])
    scale, radius = SCALES[task], 0.03 * SCALES[task]
    result = []
    for sign in (-1, 1):
        chosen = None
        for fraction in (0.5, 0.35, 0.65, 0.2, 0.8):
            i = int(fraction * (len(shapes) - 1))
            delta = refs[min(i + 5, len(refs) - 1)] - refs[max(0, i - 5)]
            normal = np.array([-delta[1], delta[0], 0.0])
            if np.linalg.norm(normal) < 1e-8 * scale:
                continue
            normal /= np.linalg.norm(normal)
            for offset in (0.08, 0.12, 0.18, 0.25):
                center = refs[i] + sign * offset * scale * normal
                hazard = {"center": center.tolist(), "radius": radius}
                margins = [clearance(shape, hazard) for shape in shapes]
                if min(margins) >= 0.005 * scale:
                    chosen = dict(
                        hazard, recorded_min_clearance=min(margins), construction_index=i, side=sign
                    )
                    break
            if chosen:
                break
        result.append(chosen)
    if all(result) and np.allclose(result[0]["center"], result[1]["center"]):
        raise ValueError("Hazard variants coincide")
    return result


def contexts(task, rows, states, split, infeasible=None):
    """Frozen contexts. Infeasible hazard contexts are omitted and listed in `infeasible`."""
    if split == "final_reserved":
        raise ValueError("This development implementation cannot open the final panel")
    result = []
    for index, row in enumerate(rows):
        pair = hazards(task, states[row["global_start"] : row["global_goal"] + 1])
        variants = [("clean", None), ("hazard0", pair[0]), ("hazard1", pair[1])]
        if split not in ("gate", "deployment"):
            variants = [variants[0], variants[1 + index % 2]]
        for condition, hazard in variants:
            if condition != "clean" and hazard is None:
                if infeasible is not None:
                    infeasible.append(
                        {
                            "split": split,
                            "source_group": row["episode"],
                            "condition": condition,
                            "id": row["record_id"] + "-" + condition,
                            "reason": "no frozen placement clears the whole recorded path",
                        }
                    )
                continue
            result.append(
                {
                    "id": row["record_id"] + "-" + condition,
                    "row": row,
                    "source_group": row["episode"],
                    "condition": condition,
                    "hazard": hazard,
                    "action_cap": CAPS[task],
                }
            )
    return result
