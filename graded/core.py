"""Graded-hazard study (9 Nov 2026): the planner-agnostic adapter core.

Everything that evolution searches lives here, so every environment runs the same maths. An environment
plug-in only supplies raw geometry in its own units (pixels, metres) plus the frozen step, and maps the
adapter's (along factor, sideways steps) back to actions with its body map.

Plan: docs/research/reports/Graded hazard study plan.md, sections 5 and 6, as cut for the 4-hour run
(graded/PREREG.md).

Genes are searched in a normalised box x in [-1, 1]^n. Gene i decodes linearly to [lo_i, hi_i]. The box
centre (x = 0) is exactly native: every trigger weight and bias decodes to 0, so the trigger score is 0, and the
trigger opens only when the score is strictly above 0.
"""

import numpy as np

OUTER = 6.0  # steps: the adapter may act only with the hazard's contact ring within this distance
SENSES = ("ahead", "side", "size", "clear", "goal", "beyond", "left", "memory")
N_SENSES = len(SENSES)
CHOOSE_FEATURES = ("clear", "safe", "lateral", "rank", "ahead", "best_clear", "left", "bias")
CHECKS = 30  # Choose's per-decision check budget, including the warm-start check


# ------------------------------------------------------------------ gene boxes

def _box(spec):
    names, lo, hi = [], [], []
    for name, a, b in spec:
        names.append(name)
        lo.append(a)
        hi.append(b)
    return names, np.array(lo, float), np.array(hi, float)


_TRIG = [("trig_" + s, -3.0, 3.0) for s in SENSES] + [("trig_bias", -3.0, 3.0)]
STEER_SPEC = (
    _TRIG
    + [("release", 0.0, 2.0)]
    + [("u_" + s, -3.0, 3.0) for s in SENSES] + [("u_bias", -3.0, 3.0)]
    + [("v_" + s, -3.0, 3.0) for s in SENSES] + [("v_bias", -3.0, 3.0)]
    + [("authority", 0.0, 1.5)]
)
CHOOSE_SPEC = (
    [("f_" + f, -3.0, 3.0) for f in CHOOSE_FEATURES]
    + [("gamma", 0.0, 5.0), ("r0", 0.0, 1.0), ("r1", 0.0, 1.0), ("exec", -1.0, 1.0)]
    + _TRIG
)
STEER_NAMES, STEER_LO, STEER_HI = _box(STEER_SPEC)
CHOOSE_NAMES, CHOOSE_LO, CHOOSE_HI = _box(CHOOSE_SPEC)
N_STEER, N_CHOOSE = len(STEER_NAMES), len(CHOOSE_NAMES)
assert N_STEER == 29 and N_CHOOSE == 21

# Run 2 (v2, 9 Nov 2026 evening): the safe direction is built in, so search tunes only when and how much.
# Steer v2 "detour": push away from the hazard by the shortfall of the travel line's side offset below the ring
# plus a margin, capped by the authority; no push once the hazard is more than `behind` steps behind.
# Choose v2 "clear first": key = z + lambda * (margin - clearance)+, and the best keyed plan is executed unless the
# planner's own plan already clears the margin (that safeguard lives in the plug-ins).
STEER2_SPEC = (
    _TRIG
    + [("release", 0.0, 2.0), ("margin", 0.0, 2.0), ("gain", 0.0, 2.0), ("authority", 0.0, 1.5),
       ("factor", 0.5, 1.2), ("behind", 0.0, 2.0)]
)
CHOOSE2_SPEC = [("margin", 0.0, 2.0), ("lam", 0.0, 10.0), ("r0", 0.0, 1.0), ("r1", 0.0, 1.0)] + _TRIG
STEER2_NAMES, STEER2_LO, STEER2_HI = _box(STEER2_SPEC)
CHOOSE2_NAMES, CHOOSE2_LO, CHOOSE2_HI = _box(CHOOSE2_SPEC)
N_STEER2, N_CHOOSE2 = len(STEER2_NAMES), len(CHOOSE2_NAMES)
assert N_STEER2 == 15 and N_CHOOSE2 == 13
BOXES = {"steer": (STEER_NAMES, STEER_LO, STEER_HI), "choose": (CHOOSE_NAMES, CHOOSE_LO, CHOOSE_HI),
         "steer2": (STEER2_NAMES, STEER2_LO, STEER2_HI), "choose2": (CHOOSE2_NAMES, CHOOSE2_LO, CHOOSE2_HI)}


def n_genes(adapter, version=1):
    return len(BOXES[adapter + ("2" if version == 2 else "")][0])


def version_of(x, adapter):
    """Genotype length tells the version: 29/21 genes = v1, 15/13 genes = v2."""
    return 2 if len(x) == len(BOXES[adapter + "2"][0]) else 1


def decode(x, kind):
    """Normalised genes x in [-1, 1]^n -> dict of named genes. kind 'steer'/'choose' accepts v1 or v2 lengths."""
    if kind in ("steer", "choose") and version_of(x, kind) == 2:
        kind = kind + "2"
    names, lo, hi = BOXES[kind]
    x = np.clip(np.asarray(x, float), -1.0, 1.0)
    if x.shape != (len(names),):
        raise ValueError(f"{kind} genotype needs {len(names)} genes, got {x.shape}")
    vals = lo + (x + 1.0) / 2.0 * (hi - lo)
    return dict(zip(names, vals.tolist()))


# ------------------------------------------------------------------ senses

def sense_vector(raw, step):
    """Raw geometry in env units -> the 8 unit-free senses (numpy array, order SENSES).

    raw keys (env units unless noted):
      ahead   hazard centre minus body position, along the block's travel direction
      side    the same, along the left normal of the travel direction (signed; + = hazard on the left)
      size    hazard radius (or box half-size) plus body radius
      clear   the block's minimum clearance (distance - hazard radius - body radius; negative = contact)
      goal    distance from the body to its goal
      beyond  cosine between the goal direction and the hazard direction (unit-free, in [-1, 1])
      left    fraction of the episode's decision budget still left (unit-free, in [0, 1])
      memory  the previous decision's sideways output in steps (Steer only; 0 otherwise)
    Distances are converted to steps, clipped at the outer limit (6 steps) and divided by 6.
    """
    s = np.empty(N_SENSES)
    for i, k in enumerate(("ahead", "side", "size", "clear", "goal")):
        s[i] = np.clip(float(raw[k]) / step, -OUTER, OUTER) / OUTER
    s[5] = float(np.clip(raw["beyond"], -1.0, 1.0))
    s[6] = float(np.clip(raw["left"], 0.0, 1.0))
    s[7] = float(np.clip(raw.get("memory", 0.0), -1.5, 1.5)) / 1.5
    return s


def within_outer(raw, step):
    """The contract's outer limit: the contact ring is within 6 steps of the body."""
    ring = np.hypot(float(raw["ahead"]), float(raw["side"])) - float(raw["size"])
    return ring <= OUTER * step


def trigger_score(g, s):
    """Linear trigger over all 8 senses. The side sense enters as |side|, so a mirrored case triggers alike."""
    s = np.asarray(s, float).copy()
    s[1] = abs(s[1])
    w = np.array([g["trig_" + k] for k in SENSES])
    return float(w @ s + g["trig_bias"])


def gate(g, s, was_open, outer_ok):
    """Hysteresis gate. Returns (open, score). Outside the outer limit it is always shut."""
    if not outer_ok:
        return False, None
    score = trigger_score(g, s)
    if was_open:
        return score > -g.get("release", 0.0), score
    return score > 0.0, score


# ------------------------------------------------------------------ Steer

def steer_move(g, s):
    """Steer's linear move. Returns (along_factor, sideways_steps).

    sideways = A * tanh(u), in steps along the left normal of the planner's own travel direction.
    along factor = 1 + 1.5 tanh(v) for v < 0 (down to -0.5: stop or back off), 1 + 0.2 tanh(v) for v >= 0
    (up to 1.2). At v = 0 it is exactly the planner's own speed.
    """
    s = np.asarray(s, float)
    u = float(np.array([g["u_" + k] for k in SENSES]) @ s + g["u_bias"])
    v = float(np.array([g["v_" + k] for k in SENSES]) @ s + g["v_bias"])
    tv = np.tanh(v)
    factor = 1.0 + 1.5 * tv if v < 0 else 1.0 + 0.2 * tv
    sideways = g["authority"] * np.tanh(u)
    return float(factor), float(sideways)


def steer_move2(g, s):
    """Steer v2 'detour'. Returns (along_factor, sideways_steps), sideways along the left normal of travel.

    need = margin + size - |side| (steps): how far the travel line must shift for the body to pass outside the
    contact ring plus the margin; also at least margin - clear for the planner's own block. The push is
    -sign(side) * min(authority, gain * need), i.e. always away from the hazard. No push, and no slow-down, once
    the hazard is more than `behind` steps behind the body, or when nothing is needed.
    """
    s = np.asarray(s, float)
    ahead, side, size, clear = (float(v) * OUTER for v in s[:4])
    if ahead < -g["behind"]:
        return 1.0, 0.0
    need = max(g["margin"] + size - abs(side), g["margin"] - clear)
    if need <= 0.0:
        return 1.0, 0.0
    away = -1.0 if side >= 0.0 else 1.0
    return float(g["factor"]), float(away * min(g["authority"], g["gain"] * need))


class SteerState:
    """Per-episode Steer state: gate hysteresis and the memory sense."""

    def __init__(self, x):
        self.g = decode(x, "steer")
        self.version = version_of(x, "steer")
        self.open = False
        self.memory = 0.0
        self.opened = 0
        self.decisions = 0

    def decide(self, raw, step):
        """raw: env geometry for the planner's proposed block. Returns (act, factor, sideways, record)."""
        raw = dict(raw, memory=self.memory)
        s = sense_vector(raw, step)
        ok = within_outer(raw, step)
        self.open, score = gate(self.g, s, self.open, ok)
        self.decisions += 1
        rec = {"open": int(self.open), "score": score, "senses": s.round(4).tolist()}
        if not self.open:
            self.memory = 0.0
            return False, 1.0, 0.0, rec
        self.opened += 1
        factor, sideways = (steer_move2 if self.version == 2 else steer_move)(self.g, s)
        self.memory = sideways
        rec.update(factor=factor, sideways=sideways)
        return True, factor, sideways, rec


def steer_never_opens(x, logged):
    """Free outcome: replay a native run's logged senses (proposal block) through the trigger.

    logged: list of (senses list or None when outside the outer limit) per decision, from a native run with the same
    hazard. Memory is 0 on a native run until the first opening, and nothing changes before it, so if the trigger
    never opens on the native senses the run IS the native run, bit for bit.
    """
    g = decode(x, "steer")
    for s in logged:
        if s is None:
            continue
        s = np.asarray(s, float).copy()
        s[7] = 0.0
        if trigger_score(g, s) > 0.0:
            return False
    return True


# ------------------------------------------------------------------ Choose

class ChooseState:
    """Per-episode Choose state. The trigger reads the senses of the warm-start block."""

    def __init__(self, x):
        self.g = decode(x, "choose")
        self.version = version_of(x, "choose")
        self.open = False
        self.opened = 0
        self.decisions = 0
        r0, r1 = sorted((self.g["r0"], self.g["r1"]))
        self.r0, self.r1 = r0, r1
        # v2 always executes its best keyed plan, unless the planner's own plan already clears the margin
        # (checked by the plug-in with one extra check, so v2 keys at most CHECKS - 2 candidates).
        self.exec_keyed = True if self.version == 2 else self.g["exec"] > 0.0
        self.margin_steps = self.g["margin"] if self.version == 2 else None
        self.budget = CHECKS - 2 if self.version == 2 else CHECKS - 1

    def gate(self, raw_warm, step):
        s = sense_vector(dict(raw_warm, memory=0.0), step)
        ok = within_outer(raw_warm, step)
        self.open, score = gate(self.g, s, self.open, ok)
        self.decisions += 1
        if self.open:
            self.opened += 1
        return self.open, {"open": int(self.open), "score": score, "senses": s.round(4).tolist()}

    def keyed_rounds(self, n_rounds):
        """Round indices that are keyed: [round(r0*(R-1)), round(r1*(R-1))], at most 29 rounds."""
        a = int(round(self.r0 * (n_rounds - 1)))
        b = int(round(self.r1 * (n_rounds - 1)))
        rounds = list(range(a, b + 1))[: self.budget]
        return rounds

    def checks_per_round(self, n_rounds):
        """29 checks split evenly over the keyed rounds, at least 1 each (remainder to the earliest rounds)."""
        rounds = self.keyed_rounds(n_rounds)
        base, extra = divmod(self.budget, len(rounds))
        return {r: base + (1 if i < extra else 0) for i, r in enumerate(rounds)}

    def keys(self, z, feats):
        """Blended key for checked candidates: key = z - gamma * tanh(f), lower = preferred.

        z: the planner's own standardised cost for each checked candidate (lower = better).
        feats: array (n, 7) in CHOOSE_FEATURES order without the bias, already unit-free:
          clear (steps, clipped [-6, 6] / 6), safe (safe-step share in [0, 1]), lateral (steps / 6, signed away
          from the hazard), rank (planner rank / (n - 1)), ahead (steps / 6), best_clear (this decision's best checked
          clearance so far, steps / 6), left (fraction of budget left).
        """
        if self.version == 2:
            clear_steps = OUTER * np.asarray(feats, float)[:, 0]
            return np.asarray(z, float) + self.g["lam"] * np.maximum(0.0, self.g["margin"] - clear_steps)
        w = np.array([self.g["f_" + f] for f in CHOOSE_FEATURES[:-1]])
        f = np.asarray(feats, float) @ w + self.g["f_bias"]
        return np.asarray(z, float) - self.g["gamma"] * np.tanh(f)


def choose_never_opens(x, logged_warm):
    """Free outcome for Choose: replay the native warm-start senses through Choose's trigger."""
    g = decode(x, "choose")
    for s in logged_warm:
        if s is None:
            continue
        if trigger_score(g, np.asarray(s, float)) > 0.0:
            return False
    return True


# ------------------------------------------------------------------ geometry helpers shared by plug-ins

def frame(direction, fallback):
    """Unit travel direction t and its left normal n in the plane (2-D)."""
    d = np.asarray(direction, float)[:2]
    nd = np.linalg.norm(d)
    t = d / nd if nd > 1e-9 else np.asarray(fallback, float)[:2] / max(np.linalg.norm(fallback[:2]), 1e-9)
    return t, np.array([-t[1], t[0]])


def planar_raw(pos, disp, centre, size, clear, goal, left, fallback_dir):
    """Raw senses for a planar body (TwoRoom, PushT, Reacher fingertip, basketball gripper in the horizontal plane).

    pos, disp, centre, goal: positions/displacement in env units (first two coordinates used for the frame).
    """
    pos = np.asarray(pos, float)
    t, n = frame(disp, fallback_dir)
    h = np.asarray(centre, float)[:2] - pos[:2]
    to_goal = np.asarray(goal, float)[:2] - pos[:2]
    gn, hn = np.linalg.norm(to_goal), np.linalg.norm(h)
    beyond = float(to_goal @ h / (gn * hn)) if gn > 1e-9 and hn > 1e-9 else 0.0
    return {
        "ahead": float(h @ t), "side": float(h @ n), "size": float(size), "clear": float(clear),
        "goal": float(gn), "beyond": beyond, "left": float(left),
    }, t, n
