"""Render the Meta-World basketball scene with the four keep-out boxes of one case group (paper levels figure).

  python -m graded.render_basketball --trace RESULT.json --panels graded/panels/basketball.json --group s6028 --out DIR

Replays the recorded planner-alone run (trace["actions"], same seed) so the arm, ball and hoop are the real scene,
checks the replayed gripper path against the recorded one, then renders candidate frames x cameras with the boxes
(translucent, level colours; Graze as an outline) and the gripper path (a dark tube) added to the MuJoCo scene. For each image it
also writes the pixel position of every box centre (pinhole projection from the scene's GL camera), verified by
rendering a marker sphere at each centre and checking the pixel colour.
"""

import argparse
import json
import os
import sys
import types
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
try:
    import torch  # noqa: F401  (systems imports it at module level; unused here)
except ImportError:
    sys.modules["torch"] = types.ModuleType("torch")

import mujoco
import numpy as np
from PIL import Image

from residual_tdmpc2 import systems

LEVEL_RGB = {1: "#f0a09f", 2: "#e66767", 3: "#c93434", 4: "#7d1d1d"}
LEVEL_ALPHA = {1: 0.55, 2: 0.55, 3: 0.6, 4: 0.13}


def rgba(hexcol, a):
    h = hexcol.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)] + [a], dtype=np.float32)


def add_geom(scn, kind, size, pos, col):
    if scn.ngeom >= scn.maxgeom:
        return
    mujoco.mjv_initGeom(scn.geoms[scn.ngeom], kind, np.asarray(size, dtype=np.float64), np.asarray(pos, dtype=np.float64),
                        np.eye(3).flatten(), col)
    scn.ngeom += 1


def project(scn, p, W, H):
    c = mujoco.mjv_averageCamera(scn.camera[0], scn.camera[1])  # mono render = average of the stereo pair
    pos, fwd, up = np.array(c.pos), np.array(c.forward), np.array(c.up)
    right = np.cross(fwd, up)
    d = np.asarray(p) - pos
    zc = d @ fwd
    xn, yn = (d @ right) * c.frustum_near / zc, (d @ up) * c.frustum_near / zc
    top, bot = c.frustum_top, c.frustum_bottom
    halfw = (top - bot) / 2 * W / H
    col = (xn - (c.frustum_center - halfw)) / (2 * halfw) * W
    row = (1 - (yn - bot) / (top - bot)) * H
    return float(col), float(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--panels", required=True)
    ap.add_argument("--group", default="s6028")
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", default="22,26,30,36,99")
    ap.add_argument("--cams", default="160:-14:0.80,150:-20:0.85,180:-8:0.80,135:-22:0.95")
    ap.add_argument("--lookat", default="0.07,0.71,0.22")
    ap.add_argument("--path-from", type=int, default=8)  # samples 0-7: hand snapping onto the mocap after reset
    ap.add_argument("--W", type=int, default=960)
    ap.add_argument("--H", type=int, default=720)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tr = json.load(open(args.trace))["trace"]
    cases = [c for c in json.load(open(args.panels)) if c["group"] == args.group]
    boxes = {c["level"]: (np.array(c["hazard"]["box_lo"]), np.array(c["hazard"]["box_hi"])) for c in cases}
    seed = int(cases[0]["seed"])

    s = systems.make_system("mw-basketball")
    s.reset(seed)
    model, data = s.env.model, s.env.data
    model.vis.global_.offwidth, model.vis.global_.offheight = max(args.W, 640), max(args.H, 480)
    # Grey the scene itself (materials, plain geoms, textures; sites hidden) so the hazard boxes carry the only colour,
    # like the desaturated frames of the other panels. Colours only: physics is untouched.
    def wash(rgb, w=0.72, keep=0.10):
        grey = rgb.mean(-1, keepdims=True)
        return w * (keep * rgb + (1 - keep) * grey) + (1 - w)
    model.mat_rgba[:, :3] = wash(model.mat_rgba[:, :3])
    model.geom_rgba[:, :3] = wash(model.geom_rgba[:, :3])
    model.site_rgba[:, 3] = 0.0
    tex = model.tex_data.reshape(-1, 3).astype(float) / 255
    model.tex_data[:] = (wash(tex) * 255).round().astype(np.uint8).reshape(-1)
    renderer = mujoco.Renderer(model, args.H, args.W)
    frames = sorted(int(f) for f in args.frames.split(","))
    snaps = {}
    for k, a in enumerate(tr["actions"]):
        s.step(np.asarray(a))
        if k in frames:
            snaps[k] = (data.qpos.copy(), data.qvel.copy(), data.mocap_pos.copy(), data.mocap_quat.copy())
    rec, rep = np.array(tr["samples"]), np.array([x["eef"] for x in s.samples])
    n = min(len(rec), len(rep))
    dev = float(np.abs(rec[:n] - rep[:n]).max())
    print(f"replay: {len(tr['actions'])} decisions, max |eef - recorded| = {dev * 1000:.2f} mm", flush=True)

    meta = {"replay_max_dev_m": dev, "seed": seed, "group": args.group, "images": []}
    W, H = args.W, args.H
    INK = rgba("#0b0b0b", 1.0)

    def edges(scn, lo, hi, col, width):
        c = [np.array([hi[0] if i & 1 else lo[0], hi[1] if i & 2 else lo[1], hi[2] if i & 4 else lo[2]]) for i in range(8)]
        for i in range(8):
            for b in (1, 2, 4):
                if not i & b:
                    add_conn(scn, c[i], c[i | b], col, width)

    def add_conn(scn, p, q, col, width):
        if scn.ngeom >= scn.maxgeom:
            return
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.eye(3).flatten(), col)
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, width, np.asarray(p, float), np.asarray(q, float))
        scn.ngeom += 1

    for k in frames:
        qpos, qvel, mpos, mquat = snaps[k]
        data.qpos[:], data.qvel[:], data.mocap_pos[:], data.mocap_quat[:] = qpos, qvel, mpos, mquat
        mujoco.mj_forward(model, data)
        upto = rec[args.path_from:min(len(rec), 1 + 2 * (k + 1))]
        for cam_s in args.cams.split(","):
            az, el, dist = (float(v) for v in cam_s.split(":"))
            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.lookat[:] = [float(v) for v in args.lookat.split(",")]
            cam.azimuth, cam.elevation, cam.distance = az, el, dist

            renderer.update_scene(data, camera=cam)
            scn = renderer.scene
            for i in range(len(upto) - 1):
                add_conn(scn, upto[i], upto[i + 1], INK, 0.0028)
            add_geom(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.008, 0, 0], upto[0], INK)
            for lvl in (4, 1, 3):
                lo, hi = boxes[lvl]
                add_geom(scn, mujoco.mjtGeom.mjGEOM_BOX, (hi - lo) / 2, (hi + lo) / 2, rgba(LEVEL_RGB[lvl], LEVEL_ALPHA[lvl]))
                edges(scn, lo, hi, rgba(LEVEL_RGB[lvl], 1.0), 0.0016 if lvl < 4 else 0.0022)
            lo, hi = boxes[2]
            edges(scn, lo, hi, rgba(LEVEL_RGB[2], 1.0), 0.0022)  # Graze overlaps On the path: outline only
            img = renderer.render().copy()
            proj = {lvl: project(scn, (boxes[lvl][0] + boxes[lvl][1]) / 2, W, H) for lvl in boxes}
            corners = {lvl: [project(scn, [hi[0] if i & 1 else lo[0], hi[1] if i & 2 else lo[1], hi[2] if i & 4 else lo[2]], W, H)
                             for i in range(8)] for lvl, (lo, hi) in boxes.items()}
            start_px = project(scn, upto[0], W, H)
            goal_px = project(scn, tr["goal"], W, H)
            # marker-only render to verify the projection
            renderer.update_scene(data, camera=cam)
            for lvl in (1, 2, 3, 4):
                lo, hi = boxes[lvl]
                add_geom(renderer.scene, mujoco.mjtGeom.mjGEOM_SPHERE, [0.006, 0, 0], (hi + lo) / 2, np.array([0, 1, 0, 1], np.float32))
            mimg = renderer.render()
            checks = {}
            for lvl, (cx, cy) in proj.items():
                x, y = int(round(cx)), int(round(cy))
                if 0 <= x < W and 0 <= y < H:
                    px = mimg[max(0, y - 2):y + 3, max(0, x - 2):x + 3].reshape(-1, 3).astype(int)
                    checks[lvl] = bool(((px[:, 1] > 150) & (px[:, 0] < 120) & (px[:, 2] < 120)).any())
            name = f"bb-f{k:02d}-az{int(az)}-el{int(el)}-d{int(dist * 100)}"
            Image.fromarray(img).save(out / f"{name}.png")
            meta["images"].append({"file": f"{name}.png", "frame": k, "azimuth": az, "elevation": el, "distance": dist,
                                   "lookat": args.lookat, "centres_px": proj, "corners_px": corners, "start_px": start_px, "goal_px": goal_px,
                                   "marker_check": checks, "W": W, "H": H})
            print(name, "marker check", checks, flush=True)
    (out / "render_meta.json").write_text(json.dumps(meta, indent=1))
    print("RENDER DONE", flush=True)


if __name__ == "__main__":
    main()
