"""Named-body geometry in physical coordinates, independent of rendered images.

Cube's 2 cm effector ball is an explicitly defined protected volume centred at
the pinch site, not an envelope for the complete gripper or arm. Other dimensions
come from the pinned environment definitions and are checked against live models.
"""

import numpy as np

SCALES = {"tworoom": 224.0, "pusht": 512.0, "reacher": 0.24, "cube": 0.5}
T_POLYGONS = (
    np.array([[-60, 30], [60, 30], [60, 0], [-60, 0]], dtype=float),
    np.array([[-15, 30], [-15, 120], [15, 120], [15, 30]], dtype=float),
)


def xyz(point):
    point = np.asarray(point, dtype=float)
    return np.pad(point, (0, 3 - len(point)))


def rotation(quaternion):
    q = np.asarray(quaternion, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-12:
        raise ValueError("Invalid wxyz orientation")
    w, x, y, z = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def segment_distance(point, start, end):
    delta = end - start
    amount = np.clip(np.dot(point - start, delta) / max(np.dot(delta, delta), 1e-30), 0, 1)
    return float(np.linalg.norm(point - start - amount * delta))


def polygon_distance(point, polygon):
    """Signed distance to one convex polygon, either winding order."""
    edges = np.roll(polygon, -1, axis=0) - polygon
    relative = point - polygon
    cross = edges[:, 0] * relative[:, 1] - edges[:, 1] * relative[:, 0]
    inside = np.all(cross >= -1e-12) or np.all(cross <= 1e-12)
    distance = min(
        segment_distance(point, a, b) for a, b in zip(polygon, np.roll(polygon, -1, axis=0))
    )
    return -distance if inside else distance


def convex_hull(points):
    """Deterministic planar monotone-chain hull for the swept T components."""
    points = sorted(set(tuple(p) for p in points))

    def cross(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    def chain(sequence):
        result = []
        for point in sequence:
            while len(result) >= 2 and cross(result[-2], result[-1], point) <= 0:
                result.pop()
            result.append(point)
        return result

    return np.asarray(chain(points)[:-1] + chain(reversed(points))[:-1])


def recorded(task, state):
    """Public state -> geometry. No recorded actions or future live states used."""
    state = np.asarray(state, dtype=float).reshape(-1)
    if not np.isfinite(state).all():
        raise ValueError("Nonfinite physical state")
    if task == "tworoom":
        return {
            "reference": xyz(state[:2]),
            "balls": [(xyz(state[:2]), 7.0)],
            "polygons": [],
            "boxes": [],
        }
    if task == "pusht":
        angle = state[4]
        matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        return {
            "reference": xyz(state[:2]),
            "balls": [(xyz(state[:2]), 15.0)],
            "polygons": [p @ matrix.T + state[2:4] for p in T_POLYGONS],
            "polygon_origin": state[2:4].copy(),
            "polygon_angle": float(angle),
            "boxes": [],
        }
    if task == "reacher":
        angles = np.array([state[0], state[0] + state[1]])
        point = 0.12 * np.array([np.cos(angles).sum(), np.sin(angles).sum()])
        return {
            "reference": xyz(point),
            "balls": [(xyz(point), 0.01)],
            "polygons": [],
            "boxes": [],
            "ball_joint_angles": state[:2].copy(),
            "ball_reach_bound": 0.24,
        }
    if task == "cube":
        if state.shape != (28,):
            raise ValueError("Expected original 28-dimensional Cube observation")
        shift = np.array([0.425, 0, 0])
        effector, cube = state[12:15] / 10 + shift, state[19:22] / 10 + shift
        return {
            "reference": effector,
            "balls": [(effector, 0.02)],
            "polygons": [],
            "boxes": [(cube, rotation(state[22:26]), np.full(3, 0.02))],
        }
    raise ValueError("Unknown geometry task")


def clearance(shape, hazard):
    if hazard is None:
        return 1.0  # Clean feature sentinel is specified separately in normalized features.
    center, radius = np.asarray(hazard["center"], dtype=float), float(hazard["radius"])
    distances = [np.linalg.norm(c - center) - r - radius for c, r in shape["balls"]]
    distances.extend(polygon_distance(center[:2], p) - radius for p in shape["polygons"])
    for c, matrix, half in shape["boxes"]:
        delta = np.abs(matrix.T @ (center - c)) - half
        signed = np.linalg.norm(np.maximum(delta, 0)) + min(float(np.max(delta)), 0)
        distances.append(signed - radius)
    return float(min(distances))


def swept_clearance(before, after, hazard):
    """Conservative between-sample clearance; rigid interpolation is explicit.

    Ball translation uses exact segment distance. Polygons/boxes use an endpoint
    distance minus a bound on maximum point movement over that substep. This can
    overcount violations, so both sampled and swept results are recorded.
    """
    if hazard is None:
        return 1.0
    center, radius = np.asarray(hazard["center"]), hazard["radius"]
    values = [
        segment_distance(center, a, b) - max(ra, rb) - radius
        for (a, ra), (b, rb) in zip(before["balls"], after["balls"])
    ]
    if "ball_joint_angles" in before:
        # Under linear joint interpolation, the end-point chord error is bounded
        # by max ||position''|| / 8 <= reach * (sum |joint delta|)^2 / 8.
        angles = np.abs(after["ball_joint_angles"] - before["ball_joint_angles"]).sum()
        values[0] -= before["ball_reach_bound"] * angles**2 / 8
    for a, b in zip(before["polygons"], after["polygons"]):
        # The endpoint hull contains linear vertex paths. Expand it by the
        # rotation chord error, measured about the actual body origin.
        ca = before["polygon_origin"]
        angle = after["polygon_angle"] - before["polygon_angle"]
        error = max(np.linalg.norm(a - ca, axis=1)) * angle**2 / 8
        hull = convex_hull(np.concatenate([a, b]))
        values.append(polygon_distance(center[:2], hull) - radius - error)
    for (ca, ra, ha), (cb, rb, hb) in zip(before["boxes"], after["boxes"]):
        angle = np.arccos(np.clip((np.trace(ra.T @ rb) - 1) / 2, -1, 1))
        bound = np.linalg.norm(cb - ca) + max(np.linalg.norm(ha), np.linalg.norm(hb)) * angle
        shapes = [
            {"balls": [], "polygons": [], "boxes": [box]} for box in [(ca, ra, ha), (cb, rb, hb)]
        ]
        first, last = [clearance(s, hazard) for s in shapes]
        values.append(min(first, last, (first + last - bound) / 2))
    return float(min(values))


def descriptor(points, hazard, task):
    points = np.asarray(points, dtype=float)
    scale = SCALES[task]
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    length = float(lengths.sum()) / scale
    vertical = float(np.ptp(points[:, 2])) / scale if task == "cube" else 0.0
    axis = (np.asarray(hazard["center"]) - points[0]) if hazard else np.zeros(3)
    norm = np.linalg.norm(axis[:2])
    stationary = length < 1e-6
    degenerate = stationary or norm < scale * 1e-6 or hazard is None
    side = 0.5
    if not degenerate:
        offset = points[:, :2] - points[0, :2]
        lateral = (axis[0] * offset[:, 1] - axis[1] * offset[:, 0]) / norm / scale
        signed = float(np.trapezoid(lateral)) / max(1, len(points) - 1)
        side = 1.0 if signed > 0.01 else 0.0 if signed < -0.01 else 0.5
    return {
        "values": [side, min(length / 4, 1), min(vertical, 1)],
        "path_length_scaled": length,
        "vertical_scaled": vertical,
        "stationary": stationary,
        "degenerate_side": degenerate,
    }
