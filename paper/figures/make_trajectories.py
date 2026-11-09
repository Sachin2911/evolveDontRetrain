"""Trajectory figure for paper v2: recorded reruns of chosen test cases (graded/media.py).

  ../../../.venv/bin/python make_trajectories.py [--tags m1] [--out trajectories.pdf] [--spec spec.json]

Reads LeWM_Experiment1/graded/evidence/<box>/out/media-<tag>.json (written by graded.media on the host), the recorded
results in graded/evidence/<box>/prod/<env>/results/<job>.json and, for the AP environments, the camera frames in
graded/evidence/<box>/prod/<env>/media/<job>.npz. Hazard geometry comes from graded/panels/<env>.json.

Each panel is one test case: the native planner's path and the adapters' paths, the hazard (red), its contact
boundary (dashed ring = hazard radius + body radius), the goal, and open circles where the adapter's gate was open.
Every rerun is checked against its test outcome (media summary field same_as_test); a mismatch is printed and the
panel title says so.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2] / "LeWM_Experiment1" / "graded"
EVID = ROOT / "evidence"

plt.rcParams.update({"font.family": "sans-serif", "font.size": 6.5, "axes.linewidth": 0.5, "pdf.fonttype": 42})
INK, MUTED, GRID, HAZARD = "#0b0b0b", "#52514e", "#e6e5e0", "#d64545"
STYLE = {  # arm prefix -> (label, color, lw, dash)
    "native": ("planner alone", INK, 1.0, "-"),
    # colour = search method (palette slot 1 blue = CMA-ES, slot 2 orange = random); dash = adapter
    "v2-steer-cma": ("Steer, CMA-ES", "#2a78d6", 1.1, "-"),
    "v2-steer-rnd": ("Steer, random search", "#eb6834", 1.1, "-"),
    "v2-choose-cma": ("Choose, CMA-ES", "#2a78d6", 1.1, (0, (4, 1.6))),
    "v2-choose-rnd": ("Choose, random search", "#eb6834", 1.1, (0, (4, 1.6))),
    "v2-stack-cma": ("Choose + Steer, CMA-ES", "#2a78d6", 1.1, (0, (1.2, 1.2))),
    "v2-stack-rnd": ("Choose + Steer, random", "#eb6834", 1.1, (0, (1.2, 1.2))),
    "choose-cma": ("Choose v1, CMA-ES", "#7f7f7f", 0.9, (0, (1, 1))),
    "choose-rnd": ("Choose v1, random", "#7f7f7f", 0.9, (0, (1, 1))),
    "steer-cma": ("Steer v1, CMA-ES", "#7f7f7f", 0.9, (0, (1, 1))),
    "steer-rnd": ("Steer v1, random", "#7f7f7f", 0.9, (0, (1, 1))),
    "stack-cma": ("Stack v1, CMA-ES", "#7f7f7f", 0.9, (0, (1, 1))),
    "stack-rnd": ("Stack v1, random", "#7f7f7f", 0.9, (0, (1, 1))),
}
FRAME = {"tworoom": dict(scale=1.0), "pusht": dict(scale=224 / 512)}
HALF_RINGS = 3.2  # minimum half-width of each panel's window, in contact-ring radii
SHOW_GATES = False  # gate-open markers clutter Steer paths (its gate is open for most decisions near the hazard)
OUTCOME = []  # (case, arm label, colour, outcome word) collected per panel for the text under it
ENV_TITLE = {"tworoom": "TwoRoom", "pusht": "PushT", "reacher": "Reacher (unseen)", "basketball": "Basketball"}


def style(arm):
    for k in sorted(STYLE, key=len, reverse=True):
        if arm.startswith(k):
            return STYLE[k]
    return (arm, MUTED, 0.8, "-")


def panels_geometry():
    geo = {}
    for env in ("tworoom", "pusht", "reacher", "basketball"):
        for c in json.loads((ROOT / "panels" / f"{env}.json").read_text()):
            geo[c["id"]] = c
    return geo


def load_media(tags):
    rows = []
    for box in sorted(p.name for p in EVID.iterdir() if p.is_dir()):
        for tag in tags:
            p = EVID / box / "out" / f"media-{tag}.json"
            if p.exists():
                for r in json.loads(p.read_text()):
                    rows.append(dict(r, box=box))
    return rows


def test_outcomes():
    per = {}
    for p in EVID.glob("*/out/test-*.json"):
        if p.name == "test-pre.json":
            continue
        for arm, rows in json.loads(p.read_text())["per_case"].items():
            per.setdefault(arm, {}).update({r["case_id"]: r for r in rows})
    return per


TESTS = None


def same_as_test(row, o):
    global TESTS
    if TESTS is None:
        TESTS = test_outcomes()
    ref = TESTS.get(row["arm"], {}).get(row["case"])
    if ref is None:
        return row.get("same_as_test")
    return all(bool(o.get(k)) == bool(ref.get(k)) for k in ("success", "violation", "safe_completion"))


def result(row):
    env = row["case"].split("-")[0]
    return json.loads((EVID / row["box"] / "prod" / env / "results" / f"{row['job']}.json").read_text())


def draw_path(ax, p, color, lw, ls, label):
    """Path with a white halo (so crossing paths stay readable), a start dot and an end tick."""
    ax.plot(p[:, 0], p[:, 1], color="white", lw=lw + 1.8, solid_capstyle="round", zorder=4)
    ax.plot(p[:, 0], p[:, 1], color=color, lw=lw, ls=ls, solid_capstyle="round", zorder=5, label=label)
    ax.plot(p[0, 0], p[0, 1], marker="o", ms=3.2, color=color, mec="white", mew=0.5, zorder=7)
    ax.plot(p[-1, 0], p[-1, 1], marker="s", ms=2.8, color=color, mec="white", mew=0.5, zorder=7)


def gate_points(o, positions, per_decision):
    """Positions where the gate was open (Steer or Choose), one marker per open decision."""
    recs = (o.get("steer_records") or []) or (o.get("choose_records") or [])
    idx = [k * per_decision for k, r in enumerate(recs) if r.get("open")]
    idx = [i for i in idx if i < len(positions)]
    return positions[idx] if idx else np.zeros((0, 2))


def outcome_word(o):
    if not o.get("completed"):
        return "error"
    return ("safe finish" if o.get("safe_completion") else ("touched" if o.get("violation") else "clear")) + \
        ("" if o.get("success") or o.get("safe_completion") else ", no finish")


def ap_panel(ax, case, rows, geo, title):
    env = case.split("-")[0]
    g = geo[case]
    scale = FRAME.get(env, {}).get("scale")
    nat = next(r for r in rows if r["arm"] == "native")
    if scale is not None:
        z = np.load(EVID / nat["box"] / "prod" / env / "media" / f"{nat['job']}.npz")
        frame = z["frames"][0].astype(float) / 255.0
        frame = 0.35 * frame + 0.65 * frame.mean(-1, keepdims=True)
        ax.imshow(frame, extent=(0, frame.shape[1], frame.shape[0], 0), interpolation="lanczos", alpha=0.6)
    else:
        scale = 1.0
    c = np.asarray(g["hazard"]["center"][:2]) * scale
    r = float(g["hazard"]["radius"]) * scale
    ring = (float(g["hazard"]["radius"]) + float(g["body_radius"])) * scale
    ax.add_patch(plt.Circle(c, r, color=HAZARD, alpha=0.55, lw=0, zorder=3))
    ax.add_patch(plt.Circle(c, ring, fill=False, ec=HAZARD, lw=0.7, ls=(0, (2.2, 1.6)), zorder=3))
    allp = []
    for row in rows:
        o = result(row)
        p = np.asarray(o.get("positions") or [], float)[:, :2] * scale
        if not len(p):
            continue
        allp.append(p)
        label, color, lw, ls = style(row["arm"])
        draw_path(ax, p, color, lw, ls, f"{label}: {outcome_word(o)}")
        OUTCOME.append((case, label, color, ls, outcome_word(o)))
        if SHOW_GATES and row["arm"] != "native":
            gp = gate_points(o, p, 5)
            ax.scatter(gp[:, 0], gp[:, 1], s=7, facecolor="white", edgecolor=color, lw=0.6, zorder=6)
        inside = np.flatnonzero(np.linalg.norm(p - c, axis=1) < ring)
        if inside.size:
            q = p[inside[0]]
            ax.plot(q[0], q[1], marker="x", ms=4.5, mew=1.1, color=HAZARD, zorder=8)
        print(f"  {case} {row['arm']}: {outcome_word(o)}, min clear {o.get('min_clear_steps'):+.2f} steps, "
              f"rerun matches test: {same_as_test(row, o)}")
    goal = np.asarray(g["goal"][:2]) * scale
    ax.scatter([goal[0]], [goal[1]], marker="*", s=40, color="#f2c94c", edgecolor=INK, lw=0.4, zorder=6)
    reach = max(float(np.min(np.linalg.norm(q - c, axis=1))) for q in allp) if allp else 0.0
    half = max(HALF_RINGS * ring, reach + 1.3 * ring)  # around the hazard, wide enough for every closest approach
    mid = c
    ax.set_xlim(mid[0] - half, mid[0] + half)
    if scale is not None and env in FRAME:
        ax.set_ylim(mid[1] + half, mid[1] - half)  # image coordinates: y down
    else:
        ax.set_ylim(mid[1] - half, mid[1] + half)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.set_title(title, fontsize=6.5, color=INK, pad=2)


def td_panel(ax, case, rows, geo, title):
    """Basketball, top view (x, y in metres): gripper path and the keep-out box's footprint."""
    g = geo[case]
    lo, hi = np.asarray(g["hazard"]["box_lo"]), np.asarray(g["hazard"]["box_hi"])
    ax.add_patch(plt.Rectangle(lo[:2], *(hi - lo)[:2], color=HAZARD, alpha=0.45, lw=0, zorder=3))
    allp = []
    for row in rows:
        o = result(row)
        tr = o.get("trace") or {}
        p = np.asarray(tr.get("tcp") or [], float)
        if not len(p):
            continue
        p3 = p
        p = p[:, :2]
        allp.append(p)
        inside = np.flatnonzero(np.all((p3 >= lo) & (p3 <= hi), axis=1))
        if inside.size:
            ax.plot(p[inside[0], 0], p[inside[0], 1], marker="x", ms=4.5, mew=1.1, color=HAZARD, zorder=8)
        label, color, lw, ls = style(row["arm"])
        draw_path(ax, p, color, lw, ls, f"{label}: {outcome_word(o)}")
        OUTCOME.append((case, label, color, ls, outcome_word(o)))
        if SHOW_GATES and row["arm"] != "native":
            logs = (tr.get("steer") or []) or (tr.get("choose") or [])
            ts = [int(r.get("t", k)) for k, r in enumerate(logs) if r.get("open")]
            ts = [t for t in ts if t < len(p)]
            if ts:
                ax.scatter(p[ts, 0], p[ts, 1], s=7, facecolor="white", edgecolor=color, lw=0.6, zorder=5)
        print(f"  {case} {row['arm']}: {outcome_word(o)}, min clear {o.get('min_clear_steps'):+.2f} steps, "
              f"rerun matches test: {same_as_test(row, o)}")
    goal = np.asarray(g["goal"][:2])
    ax.scatter([goal[0]], [goal[1]], marker="*", s=40, color="#f2c94c", edgecolor=INK, lw=0.4, zorder=6)
    step = float(g["hazard"]["step_m"])
    half = 2 * HALF_RINGS * step
    mid = (lo[:2] + hi[:2]) / 2
    ax.set_xlim(mid[0] - half, mid[0] + half)
    ax.set_ylim(mid[1] - half, mid[1] + half)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.set_title(title, fontsize=6.5, color=INK, pad=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="m1")
    ap.add_argument("--out", default=str(HERE / "trajectories.pdf"))
    ap.add_argument("--cases", default="", help="comma list of case ids in panel order (default: every recorded case)")
    ap.add_argument("--arms", default="", help="comma list of arms to draw (default: every recorded arm)")
    args = ap.parse_args()
    geo = panels_geometry()
    rows = load_media(args.tags.split(","))
    if not rows:
        raise SystemExit("no media summaries found")
    cases = args.cases.split(",") if args.cases else list(dict.fromkeys(r["case"] for r in rows))
    arms = set(args.arms.split(",")) if args.arms else None
    n = len(cases)
    fig, axes = plt.subplots(1, n, figsize=(1.75 * n, 2.0), gridspec_kw=dict(wspace=0.06))
    axes = np.atleast_1d(axes)
    handles = {}
    for ax, case in zip(axes, cases):
        env = case.split("-")[0]
        rs = [r for r in rows if r["case"] == case and (arms is None or r["arm"] in arms or r["arm"] == "native")]
        lvl = case.split("-")[2]
        title = f"{ENV_TITLE[env]}, level {lvl[1:]}"
        (td_panel if env == "basketball" else ap_panel)(ax, case, rs, geo, title)
        lines = [o for o in OUTCOME if o[0] == case]
        for k, (_, lab, col, ls, word) in enumerate(lines):
            y = -0.06 - 0.075 * k
            ax.plot([0.02, 0.12], [y + 0.012, y + 0.012], transform=ax.transAxes, color=col, lw=1.1, ls=ls,
                    clip_on=False)
            short = lab.replace(", CMA-ES", " CMA").replace(", random search", " random").replace("planner alone", "planner alone")
            ax.text(0.15, y, f"{short}: {word}", transform=ax.transAxes, fontsize=5.4, color=INK, va="bottom",
                    fontweight="bold" if word.startswith("safe") else "normal")
        for h, lab in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(lab.split(":")[0], h)
    from matplotlib.lines import Line2D
    leg = list(handles.values()) + [
        Line2D([], [], ls="", marker="x", ms=4, mew=1.1, color=HAZARD, label="first touch"),
        plt.Circle((0, 0), 1, color=HAZARD, alpha=0.45, lw=0, label="hazard"),
        Line2D([], [], color=HAZARD, lw=0.6, ls=(0, (2, 1.5)), label="contact ring"),
        Line2D([], [], ls="", marker="*", ms=6, mfc="#f2c94c", mec=INK, mew=0.4, label="goal")]
    labels = list(handles.keys()) + ["first touch", "hazard", "contact ring", "goal"]
    fig.legend(leg, labels, loc="upper center", ncol=len(leg), frameon=False, fontsize=5.8,
               handlelength=1.8, columnspacing=1.0, bbox_to_anchor=(0.5, -0.12))
    fig.savefig(args.out, bbox_inches="tight", pad_inches=0.02)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
