"""Frozen twenty-field target features and initial-only selector context."""

import numpy as np

from .geometry import SCALES, clearance, recorded

FIELDS = (
    "retrieval_distance",
    "native_rank",
    "source_current_latent",
    "successor_current_latent",
    "successor_goal_latent",
    "successor_source_latent",
    "current_clearance",
    "source_clearance",
    "successor_clearance",
    "source_current_position",
    "successor_current_position",
    "recorded_segment_clearance",
    "target_dx",
    "target_dy",
    "target_dz",
    "hazard_dx",
    "hazard_dy",
    "hazard_dz",
    "hazard_radius",
    "remaining_budget",
)


def normalized_clearance(shape, hazard, scale):
    return clearance(shape, hazard) / scale if hazard else 1.0


def candidate_features(
    task,
    sources,
    costs,
    current_latent,
    goal_latent,
    latents,
    states,
    current_shape,
    hazard,
    remaining_fraction,
):
    scale = SCALES[task]
    current_latent = np.asarray(current_latent).reshape(-1)
    goal_latent = np.asarray(goal_latent).reshape(-1)
    rows = []
    for rank, source in enumerate(sources):
        source = int(source)
        segment = [recorded(task, states[source + j]) for j in range(6)]
        first, last = segment[0], segment[-1]
        z0, z1 = np.asarray(latents[source], float), np.asarray(latents[source + 5], float)
        # Retrieval features use a frozen RMS scale of one, never test-fitted statistics.
        latent = lambda a, b: float(np.linalg.norm(a - b) / np.sqrt(len(a)))  # noqa: E731
        reference = current_shape["reference"]
        delta = (last["reference"] - reference) / scale
        hazard_delta = (np.asarray(hazard["center"]) - reference) / scale if hazard else np.zeros(3)
        row = [
            float(costs[rank]) / 192,
            rank / 7,
            latent(z0, current_latent),
            latent(z1, current_latent),
            latent(z1, goal_latent),
            latent(z1, z0),
            normalized_clearance(current_shape, hazard, scale),
            normalized_clearance(first, hazard, scale),
            normalized_clearance(last, hazard, scale),
            np.linalg.norm(first["reference"] - reference) / scale,
            np.linalg.norm(last["reference"] - reference) / scale,
            min(normalized_clearance(s, hazard, scale) for s in segment),
            *delta,
            *hazard_delta,
            hazard["radius"] / scale if hazard else 0,
            remaining_fraction,
        ]
        rows.append(row)
    result = np.asarray(rows, dtype=np.float64)
    if result.shape != (8, 20) or not np.isfinite(result).all():
        raise ValueError("Invalid twenty-field feature matrix")
    return np.clip(result, -10, 10)


def causal_stream(
    task, reference, previous_reference, previous_rank, current_clearance, remaining_fraction
):
    first = previous_reference is None
    delta = np.zeros(3) if first else (reference - previous_reference) / SCALES[task]
    return np.array(
        [
            *delta,
            0 if first else previous_rank / 7,
            float(np.linalg.norm(delta)),
            np.clip(current_clearance, -10, 10),
            remaining_fraction,
            float(not first),
        ]
    )


def selector_context(task, current_latent, goal_latent, initial_shape, goal_shape, hazard):
    # Projection seed is shared globally and independent of all evaluation labels.
    projection = np.random.default_rng(20261105).normal(0, 1 / np.sqrt(192), (192, 8))
    current = np.asarray(current_latent, dtype=float).reshape(192)
    goal = np.asarray(goal_latent, dtype=float).reshape(192)
    reference = initial_shape["reference"]
    scale = SCALES[task]
    hazard_delta = (np.asarray(hazard["center"]) - reference) / scale if hazard else np.zeros(3)
    context = np.r_[
        current @ projection,
        goal @ projection,
        (goal_shape["reference"] - reference) / scale,
        hazard_delta,
        hazard["radius"] / scale if hazard else 0,
        normalized_clearance(initial_shape, hazard, scale),
        float(hazard is not None),
    ]
    return np.clip(context, -10, 10).tolist()
