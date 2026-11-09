"""Figures for paper v2 (graded hazard study). Run from anywhere:

  python paper/figures/make_figures.py

Writes PDFs next to this file. Run-2 figures (v2 arms and states) are made only when that evidence exists.
Colours: the dataviz reference palette, in fixed slot order (native is ink grey, not a series colour):
CMA-ES = slot 1 blue, random search = slot 2 orange, untrained = slot 3 aqua; gains/losses use the blue/red
diverging pair. Identity is never colour alone: each method also has its own marker.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gdata as G  # noqa: E402

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
BLUE, ORANGE, AQUA, RED = "#2a78d6", "#eb6834", "#1baf7a", "#e34948"
HAZ = "#d03b3b"
UNTR = "#8a8984"  # untrained draws are a reference, not a search method: neutral grey, hollow
METHOD = {"cma": (BLUE, "o", "CMA-ES"), "rnd": (ORANGE, "D", "Random search"), "untrained": (UNTR, "s", "Untrained adapter")}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "pdf.fonttype": 42, "axes.spines.top": False,
    "axes.spines.right": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def save(fig, name):
    fig.savefig(HERE / name)
    plt.close(fig)
    print("wrote", name)


# ------------------------------------------------------------------ hazard levels: idea (top) and real cases (bottom)

LEVEL_COL = {1: "#f0a09f", 2: "#e66767", 3: "#c93434", 4: "#7d1d1d"}  # one red hue, light -> dark = more severe
LEVEL_TXT = {
    1: ("Clear", "2-3 steps off the path.\nNothing to do."),
    2: ("Graze", "Clips the edge by a\nquarter step. A small\nnudge clears it."),
    3: ("On the path", "Centred on the path.\nNeeds a detour of a\nstep or more."),
    4: ("On the goal", "Covers the goal.\nNo safe finish:\nstop or touch."),
}


def _halo_line(ax, x, y, color, lw, z=4, ls="-"):
    ax.plot(x, y, color="white", lw=lw + 1.6, solid_capstyle="round", zorder=z - 0.1)
    ax.plot(x, y, color=color, lw=lw, ls=ls, solid_capstyle="round", zorder=z)


def _disc(ax, c, r, ring, lvl, label=True, fs=6):
    col = LEVEL_COL[lvl]
    ax.add_patch(plt.Circle(c, r, color=col, alpha=0.85, lw=0, zorder=5))
    ax.add_patch(plt.Circle(c, ring, fill=False, ls=(0, (2.2, 1.6)), color=col, lw=0.7, zorder=5))
    if label:
        ax.text(c[0], c[1], str(lvl), ha="center", va="center", fontsize=fs, color="white", fontweight="bold", zorder=6)


def _bare(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def _basketball_levels(ax, group="s6028"):
    """Basketball: a MuJoCo render of the real scene (arm, ball, hoop) at the moment the planner alone scores in test
    case basketball-s6028-L3-on, with the four keep-out boxes of that case group drawn into the scene and the gripper's
    path as a dark tube. Made by LeWM_Experiment1/graded/render_basketball.py, which replays the recorded run (graded/media.py
    tag m2; replay within 3.1 mm of the recording) in a greyed copy of the scene, so the boxes carry the only colour.
    Graze is the On-the-path box moved a quarter step sideways, so it overlaps level 3 and is drawn as an outline.
    Badge positions come from the render's own camera projection (basketball_scene.json)."""
    from PIL import Image
    img = np.asarray(Image.open(HERE / "basketball_scene.png").convert("RGB"))
    meta = json.loads((HERE / "basketball_scene.json").read_text())["images"][0]
    H, W = img.shape[:2]
    ax.imshow(img, extent=(0, W, H, 0), interpolation="lanczos")
    cr = {int(k): np.asarray(v) for k, v in meta["corners_px"].items()}
    ce = {int(k): np.asarray(v) for k, v in meta["centres_px"].items()}
    spots = {1: np.array([ce[1][0], cr[1][:, 1].min() - 34]),                 # above its box
             2: np.array([ce[2][0] - 14, cr[2][:, 1].min() - 34]),             # above its outline
             3: np.array([cr[3][:, 0].max() + 26, ce[3][1] + 22]),             # right of its box (clear of level 2)
             4: np.array([cr[4][:, 0].max() - 90, cr[4][:, 1].min() + 55])}    # top-right inside the goal box
    for lvl, (x, y) in spots.items():
        ax.plot(x, y, marker="o", ms=7.4, color=LEVEL_COL[lvl], mec="white", mew=0.7, zorder=6)
        ax.text(x, y, str(lvl), ha="center", va="center", fontsize=6.2, color="white", fontweight="bold", zorder=7)
    gx, gy = meta["goal_px"]
    ax.plot(gx, gy, marker="*", ms=8, color="#f2c94c", mec=INK, mew=0.5, zorder=8)
    ax.set_xlim(95, 865)  # square crop: arm, boxes, path and hoop
    ax.set_ylim(815, 45)
    ax.set_aspect("equal")
    _bare(ax)
    ax.set_title("Basketball", fontsize=7, color=INK, pad=2, fontweight="bold")


def fig_levels():
    fig = plt.figure(figsize=(7.0, 3.9))
    gs = fig.add_gridspec(2, 12, height_ratios=[1.0, 1.35], hspace=0.62, wspace=0.0)
    # Top: the four placements on one idealised path (units = steps, the planner's typical move).
    t = np.linspace(0, 1, 200)
    px, py = 10 * t, 2.2 * np.sin(np.pi * t)
    anchor = np.array([5.0, 2.2])
    r, ring = 1.0, 1.35  # hazard radius = 1 step; contact ring = hazard + body
    centres = {1: anchor + [0, ring + 2.5], 2: anchor + [0, ring - 0.25], 3: anchor, 4: np.array([10.0, 0.0])}
    for lvl in (1, 2, 3, 4):
        ax = fig.add_subplot(gs[0, 3 * (lvl - 1):3 * lvl])
        _halo_line(ax, px, py, INK, 1.2)
        ax.plot(0, 0, marker="o", ms=4, color=INK, zorder=7)
        if lvl == 4:  # success region around the goal, drawn under the hazard
            ax.add_patch(plt.Circle((10, 0), 0.9, fill=False, ec=MUTED, lw=0.6, ls=(0, (1, 1.2)), zorder=4))
        _disc(ax, centres[lvl], r if lvl < 4 else 1.15, ring if lvl < 4 else 1.5, lvl, label=False)
        ax.plot(10, 0, marker="*", ms=8, color="#f2c94c", mec=INK, mew=0.5, zorder=8)
        if lvl in (1, 2):
            ax.plot(*anchor, marker="D", ms=2.6, color=INK, zorder=7)
        if lvl == 1:  # the clear gap
            ax.annotate("", xy=anchor + [0, 0.12], xytext=anchor + [0, 2.5 - 0.02], zorder=6,
                        arrowprops=dict(arrowstyle="<->", lw=0.6, color=INK2, shrinkA=0, shrinkB=0))
            ax.text(anchor[0] + 0.35, anchor[1] + 1.25, "2-3 steps", fontsize=5.6, color=INK2, va="center")
        title, body = LEVEL_TXT[lvl]
        ax.set_title(f"Level {lvl}: {title}", fontsize=7, color=INK, pad=1, fontweight="bold", loc="left")
        ax.text(0.02, -0.02, body, transform=ax.transAxes, fontsize=6, color=INK2, va="top", ha="left", linespacing=1.2)
        ax.set_xlim(-0.8, 11.8)
        ax.set_ylim(-1.7, 7.6)
        ax.set_aspect("equal")
        _bare(ax)
    # Bottom: the same four levels on one real case per system, drawn over its first camera frame.
    tasks = [("tworoom", "TwoRoom"), ("pusht", "PushT"), ("reacher", "Reacher")]
    k_reacher = (112 / np.tan(np.radians(22.5))) / 0.74
    for i, (task, title) in enumerate(tasks):
        ax = fig.add_subplot(gs[1, 3 * i:3 * i + 3])
        d = json.loads((G.LE / f"hazard_design/evidence/20261109/levels/{task}-levels.json").read_text())
        z = np.load(G.LE / d["source"], allow_pickle=True)
        if task == "tworoom":
            to_px, s = (lambda q: np.asarray(q, float)[..., :2]), 1.0
        elif task == "pusht":
            to_px, s = (lambda q: np.asarray(q, float)[..., :2] * (224 / 512)), 224 / 512
        else:
            to_px = lambda q: np.stack([112 + k_reacher * np.asarray(q, float)[..., 0],  # noqa: E731
                                        112 - k_reacher * np.asarray(q, float)[..., 1]], -1)
            s = k_reacher
        frame = z["frames"][0].astype(float) / 255.0
        grey = frame.mean(-1, keepdims=True)
        frame = 0.35 * frame + 0.65 * grey  # mostly desaturated, so the hazards carry the colour
        ax.imshow(frame, extent=(0, 224, 224, 0), alpha=0.6, interpolation="lanczos")
        path = to_px(z["positions"])
        _halo_line(ax, path[:, 0], path[:, 1], INK, 1.1)
        ax.plot(*path[0], marker="o", ms=3.6, color=INK, mec="white", mew=0.5, zorder=7)
        ax.plot(*path[-1], marker="*", ms=8, color="#f2c94c", mec=INK, mew=0.5, zorder=8)
        pts = [path]
        for name, lv in d["levels"].items():
            lvl = int(name.split("-")[0])
            c = to_px(lv["center"])
            _disc(ax, c, lv["radius"] * s, (lv["radius"] + d["body"]) * s, lvl, fs=6.2)
            pts.append(np.array([c - (lv["radius"] + d["body"]) * s, c + (lv["radius"] + d["body"]) * s]))
        allp = np.concatenate(pts)
        lo, hi = allp.min(0) - 10, allp.max(0) + 10
        half = max(hi - lo) / 2
        mid = (lo + hi) / 2
        ax.set_xlim(max(0, mid[0] - half), min(224, mid[0] + half))
        ax.set_ylim(min(224, mid[1] + half), max(0, mid[1] - half))
        ax.set_aspect("equal")
        _bare(ax)
        ax.set_title(title, fontsize=7, color=INK, pad=2, fontweight="bold")
    _basketball_levels(fig.add_subplot(gs[1, 9:12]))
    handles = [Line2D([], [], color=INK, lw=1.2, label="planner's path"),
               Line2D([], [], ls="none", marker="o", ms=3.6, color=INK, label="start"),
               Line2D([], [], ls="none", marker="*", ms=7, color="#f2c94c", mec=INK, mew=0.5, label="goal"),
               Line2D([], [], ls="none", marker="D", ms=2.8, color=INK, label="placement point"),
               Line2D([], [], color=LEVEL_COL[3], lw=0.7, ls=(0, (2.2, 1.6)), label="contact ring")]
    handles += [plt.Circle((0, 0), 1, color=LEVEL_COL[k], lw=0, label=f"level {k}") for k in (1, 2, 3, 4)]
    fig.legend(handles=handles, loc="lower center", ncol=9, frameon=False, fontsize=6, handlelength=1.5,
               columnspacing=1.0, bbox_to_anchor=(0.5, -0.04))
    save(fig, "levels.pdf")


# ------------------------------------------------------------------ search curves

def fig_search(version=1):
    st = G.states(version)
    if not st:
        return
    keys = sorted(st, key=lambda k: (k[0] != "steer", k[1]))
    fig, axs = plt.subplots(1, len(keys), figsize=(7.0, 1.55), sharey=True)
    axs = np.atleast_1d(axs)
    for ax, key in zip(axs, keys):
        gens = st[key]["generations"]
        x = [g["generation"] for g in gens]
        for arm in ("cma", "rnd"):
            col, mk, lab = METHOD[arm]
            d = np.array([[c["S"] - g["native"]["S"] for c in g[arm]] for g in gens], float)
            ax.plot(x, d.mean(1), color=col, lw=1.6, label=f"{lab}: population mean", zorder=3)
            ax.step(x, np.maximum.accumulate(d.max(1)), where="post", color=col, lw=0.8, alpha=0.9,
                    ls=(0, (3, 1.5)), label=f"{lab}: best so far", zorder=2)
        ax.axhline(0, color=INK2, lw=0.7, zorder=1)
        ax.set_title(f"{key[0].capitalize()}, repeat {key[1]}", pad=2)
        ax.set_xlabel("generation")
        ax.grid(axis="y", color=GRID, lw=0.5)
        ax.set_xlim(min(x) - 0.3, max(x) + 0.3)
    axs[0].set_ylabel("safe finishes above\nthe planner alone")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.13))
    fig.subplots_adjust(wspace=0.08)
    save(fig, "search.pdf")


# ------------------------------------------------------------------ test panel: every locked nominee and draw

def families(version):
    fam = []
    pre = "v2-" if version == 2 else ""
    for adapter, label in (("steer", "Steer"), ("choose", "Choose"), ("stack", "Choose + Steer")):
        rows = {"cma": [], "rnd": [], "untrained": []}
        for a in G.arms(version):
            i = G.arm_info(a)
            if i["adapter"] == adapter and not i["veto"] and i["method"] in rows:
                rows[i["method"]].append(a)
        if any(rows.values()):
            fam.append((label, rows))
    return fam


def fig_test():
    fams = families(2)  # the paper reports the version-2 run only
    envs = G.ENVS
    fig, axs = plt.subplots(1, len(envs), figsize=(7.0, 0.42 * len(fams) + 0.7), sharey=True)
    for ax, env in zip(axs, envs):
        nat = G.counts("native", (env,))["S"]
        ax.axvline(nat, color=INK, lw=0.8, ls=(0, (3, 2)), zorder=2.5)
        for yi, (label, rows) in enumerate(fams):
            y = len(fams) - 1 - yi
            if yi % 2 == 0:
                ax.axhspan(y - 0.5, y + 0.5, color="#f4f3f0", lw=0, zorder=0)
            for off, m in ((0.18, "cma"), (0.0, "rnd"), (-0.18, "untrained")):
                col, mk, _ = METHOD[m]
                for a in rows[m]:
                    v = G.counts(a, (env,))["S"]
                    ax.plot(v, y + off, marker=mk, ms=4.4 if mk != "D" else 3.8, color=col,
                            mfc=col if m != "untrained" else "white", mec=col if m == "untrained" else "white",
                            mew=0.6 if m != "untrained" else 0.9, ls="none", zorder=3)
        ax.set_yticks(range(len(fams)))
        ax.set_yticklabels([f[0] for f in reversed(fams)])
        ax.set_xlim(-0.5, 18.5)
        ax.set_xticks([0, 6, 12, 18])
        ax.grid(axis="x", color=GRID, lw=0.5)
        ax.set_title(f"{G.ENV_NAME[env]}{' (unseen)' if env == 'reacher' else ''}", pad=2)
        ax.text(nat, len(fams) - 0.45, f"planner alone: {nat}", ha="center", va="bottom", fontsize=6, color=INK2)
        ax.set_ylim(-0.5, len(fams) - 0.1)
    handles = [Line2D([], [], marker=METHOD[m][1], color=METHOD[m][0], mfc=METHOD[m][0] if m != "untrained" else "white",
                      ls="none", ms=4, label=METHOD[m][2]) for m in ("cma", "rnd", "untrained")]
    handles.append(Line2D([], [], color=INK, ls=(0, (3, 2)), lw=0.8, label="Planner alone"))
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.08))
    fig.supxlabel("safe finishes on the level 1-3 test cases (18 per system)", fontsize=7, color=INK2, y=-0.04)
    fig.subplots_adjust(wspace=0.06)
    save(fig, "test.pdf")


# ------------------------------------------------------------------ paired gains and losses against native

def locked_arms(version):
    out = []
    for label, rows in families(version):
        for m in ("cma", "rnd"):
            for a in sorted(rows[m]):
                out.append(a)
    return out


def short(a):
    i = G.arm_info(a)
    meth = {"cma": "CMA-ES", "rnd": "random"}[i["method"]]
    ad = {"steer": "Steer", "choose": "Choose", "stack": "Choose + Steer"}[i["adapter"]]
    return f"{ad}, {meth}, rep. {i['repeat']}"


def fig_paired(version=1):
    arms = locked_arms(version)
    if not arms:
        return
    envs = G.ENVS
    fig, axs = plt.subplots(1, len(envs), figsize=(7.0, 0.16 * len(arms) + 0.55), sharey=True)
    for ax, env in zip(axs, envs):
        for yi, a in enumerate(arms):
            y = len(arms) - 1 - yi
            p = G.paired(a, "native", (env,))
            if (yi // 4) % 2 == 0:
                ax.axhspan(y - 0.5, y + 0.5, color="#f4f3f0", lw=0, zorder=0)
            ax.barh(y, p["gain"], color=BLUE, height=0.62, zorder=3)
            ax.barh(y, -p["loss"], color=RED, height=0.62, zorder=3)
            if p["gain"]:
                ax.text(p["gain"] + 0.15, y, str(p["gain"]) + (" *" if p["p"] < 0.05 else ""), va="center",
                        fontsize=5.6, color=INK, fontweight="bold" if p["p"] < 0.05 else "normal")
            if p["loss"]:
                ax.text(-p["loss"] - 0.15, y, str(p["loss"]), va="center", ha="right", fontsize=5.6, color=INK2)
        ax.axvline(0, color=INK2, lw=0.6)
        ax.set_xlim(-4.5, 12.5)
        ax.set_xticks([-4, 0, 4, 8, 12])
        ax.grid(axis="x", color=GRID, lw=0.5)
        ax.set_title(f"{G.ENV_NAME[env]}{' (unseen)' if env == 'reacher' else ''}", pad=2)
        ax.set_yticks(range(len(arms)))
        ax.set_yticklabels([short(a) for a in reversed(arms)])
    handles = [plt.Rectangle((0, 0), 1, 1, color=BLUE, label="gained: safe where the planner alone was not"),
               plt.Rectangle((0, 0), 1, 1, color=RED, label="lost: planner alone safe, adapter not"),
               Line2D([], [], marker="$*$", color=INK, ls="none", ms=5, label="sign test p < 0.05 (bold)")]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.05))
    fig.supxlabel("level 1-3 test cases lost (left) and gained (right), each compared with the planner alone on the same case", fontsize=7,
                  color=INK2, y=-0.02)
    fig.subplots_adjust(wspace=0.06)
    save(fig, "paired.pdf")


# ------------------------------------------------------------------ diagnosis

def fig_diag(version=1):
    pre = "v2-" if version == 2 else ""
    ch = [a for a in (f"{pre}choose-cma-r1", f"{pre}choose-cma-r2", f"{pre}choose-rnd-r1", f"{pre}choose-rnd-r2") if a in G.per_case()]
    stv = [a for a in (f"{pre}steer-cma-r1", f"{pre}steer-cma-r2", f"{pre}steer-rnd-r1", f"{pre}steer-rnd-r2") if a in G.per_case()]
    if not ch and not stv:
        return
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 1.75), gridspec_kw={"width_ratios": [1.0, 1.0, 0.9]})
    envs = G.ENVS
    for ax, group, title in ((axs[0], ch, "Choose: closest approach, levels 2-3"),
                             (axs[1], stv, "Steer: closest approach, levels 2-3")):
        for yi, env in enumerate(envs):
            y = len(envs) - 1 - yi
            nat = G.clearance("native", (env,))
            ax.plot(nat, y, marker="|", ms=9, mew=1.6, color=INK2, ls="none", zorder=4)
            for a in group:
                m = G.arm_info(a)["method"]
                col, mk, _ = METHOD[m]
                v = G.clearance(a, (env,))
                off = 0.12 if G.arm_info(a)["repeat"] == 1 else -0.12
                ax.plot([nat, v], [y + off, y + off], color=col, lw=0.7, alpha=0.6, zorder=2)
                ax.plot(v, y + off, marker=mk, ms=3.6, color=col, ls="none", zorder=3)
        ax.axvspan(-3, 0, color=HAZ, alpha=0.07, lw=0)
        ax.axvline(0, color=HAZ, lw=0.6)
        ax.set_yticks(range(len(envs)))
        ax.set_yticklabels([G.ENV_NAME[e] for e in reversed(envs)])
        ax.set_xlim(-1.3, 2.0)
        ax.set_xlabel("mean closest approach (steps)")
        ax.set_title(title, pad=2)
        ax.grid(axis="x", color=GRID, lw=0.5)
    ax = axs[2]
    rows = []
    for a in ch:
        k, o = G.choose_executed(a)
        i_ = G.arm_info(a)
        rows.append((f"Choose, {'CMA-ES' if i_['method'] == 'cma' else 'random'} {i_['repeat']}", k / max(k + o, 1), "", i_["method"]))
    for a in stv:
        pushed, opened, _ = G.steer_pushed(a)
        i_ = G.arm_info(a)
        rows.append((f"Steer, {'CMA-ES' if i_['method'] == 'cma' else 'random'} {i_['repeat']}", pushed / max(opened, 1), "", i_["method"]))
    for yi, (lab, v, _, m) in enumerate(rows):
        y = len(rows) - 1 - yi
        ax.barh(y, v, color=METHOD[m][0], height=0.62, zorder=3)
        ax.text(v + 0.02, y, f"{100 * v:.0f}%", va="center", fontsize=6, color=INK2)
    n_ch = len(ch)
    if ch and stv:
        ax.axhline(len(rows) - n_ch - 0.5, color=MUTED, lw=0.5)

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in reversed(rows)])
    ax.set_xlim(0, 1.22)
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_xticklabels(["0", "50%", "100%"])
    ax.axvline(0.5, color=INK2, lw=0.5, ls=(0, (2, 2)))
    ax.set_xlabel("share of open decisions")
    ax.set_title("Gate open: Choose ran a safer plan,\nSteer pushed (share of decisions)", pad=2, fontsize=6.6)
    ax.grid(axis="x", color=GRID, lw=0.5)
    handles = [Line2D([], [], marker="|", color=INK2, ls="none", ms=7, mew=1.5, label="Planner alone")]
    handles += [Line2D([], [], marker=METHOD[m][1], color=METHOD[m][0], ls="none", ms=4, label=METHOD[m][2]) for m in ("cma", "rnd")]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.1))
    fig.subplots_adjust(wspace=0.75)
    save(fig, "diag.pdf")


# ------------------------------------------------------------------ Block Plucking pilot (v1 data, v2 palette)

def _wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def fig_block():
    """Pooled 48 fresh Hard starts. Data copied from the v1 paper's figure data (Paper/figures/data.json)."""
    d = json.loads((G.ROOT / "Paper/figures/data.json").read_text())["block"]
    n = d["episodes"]
    style = {"Planner alone": (INK, "|", INK), "Untrained adapter": (UNTR, "s", "white"), "Random search": (ORANGE, "D", ORANGE),
             "CMA-ES": (BLUE, "o", BLUE)}
    rename = {"Native": "Planner alone", "Untrained": "Untrained adapter", "Random": "Random search", "CMA-ES": "CMA-ES"}
    rows = [dict(r, label=rename[r["label"]]) for r in d["rows"]]
    fig, axs = plt.subplots(1, 2, figsize=(3.35, 1.15), sharey=True, gridspec_kw=dict(wspace=0.18))
    for ax, key, title in zip(axs, ("sustained", "violated"), (f"Sustained safe (of {n})", f"Failures (of {n})")):
        nat = next(r for r in rows if r["label"] == "Planner alone")[key]
        ax.axvline(100 * nat / n, color=INK, lw=0.7, ls=(0, (3, 2)), zorder=1)
        for y, r in zip(range(len(rows))[::-1], rows):
            col, mk, fc = style[r["label"]]
            lo, hi = _wilson(r[key], n)
            ax.plot([100 * lo, 100 * hi], [y, y], color=col, lw=1.0, solid_capstyle="round", zorder=2, alpha=0.8)
            ax.plot(100 * r[key] / n, y, marker=mk, ms=4.6 if mk != "|" else 8, mew=1.4 if mk == "|" else 0.8,
                    color=col, mfc=fc, mec=col if fc == "white" or mk == "|" else "white", ls="none", zorder=3)
            ax.text(100 * hi + 2, y, str(r[key]), va="center", fontsize=6, color=INK)
        ax.set_title(title, fontsize=7, loc="left", pad=2)
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 25, 50, 75])
        ax.grid(axis="x", color=GRID, lw=0.5)
        ax.set_xlabel("% of starts (95% interval)", fontsize=6.2)
        ax.tick_params(axis="y", length=0)
    axs[0].set_yticks(range(len(rows))[::-1])
    axs[0].set_yticklabels([r["label"] for r in rows])
    axs[0].set_ylim(-0.6, len(rows) - 0.4)
    save(fig, "block.pdf")


if __name__ == "__main__":
    # The paper reports the version-2 run only (Sachin, 9 Nov 2026): every figure reads the v2 arms and states.
    fig_block()
    fig_levels()
    fig_search(2)
    fig_test()
    fig_paired(2)
    fig_diag(2)
