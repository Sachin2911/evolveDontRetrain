"""Gated gain-and-residual on the planner's executed action block (residualRecipe).

The hook acts after Anchored Planning has chosen its five-action block. A system oracle
rolls the native block forward against the declared hazard (kinematic for TwoRoom and
PushT, reused from qd_adapters.v4hook; Reacher and Cube oracles live in
residual.oracles_extra). If the block's minimum clearance, in hazard radii, is below the
gate onset, a small head reads margin features and emits a gain and a two-direction
residual, expressed in ACTION space through the oracle's direction basis (radially away
from the hazard, and a perpendicular). The corrected block is re-rolled; if it is less
clear than the native block the correction is vetoed. With the gate closed the native
block object is returned untouched (bitwise parity).
"""

import numpy as np

from qd_adapters import v4hook

HIDDEN = 8
N_FEATURES = 7
N_OUT = 3  # gain logit, radial logit, perpendicular logit
N_WEIGHTS = N_FEATURES * HIDDEN + HIDDEN + HIDDEN * N_OUT + N_OUT  # 91
N_GENES = N_WEIGHTS + 2  # + authority c (raw), onset (raw)
GAIN_BOUND = 0.2


def structure(genes):
    """Map the two raw structure genes to authority fraction c and gate onset (hazard radii)."""
    c_raw, onset_raw = float(genes[N_WEIGHTS]), float(genes[N_WEIGHTS + 1])
    c = 0.05 + 0.45 / (1.0 + np.exp(-c_raw))            # (0.05, 0.50) of native magnitude
    onset = 0.25 + 2.75 / (1.0 + np.exp(-onset_raw))    # (0.25, 3.0) hazard radii
    return c, onset


def unpack(genes):
    g = np.asarray(genes, dtype=np.float64)
    if g.shape != (N_GENES,):
        raise ValueError("Genotype must have %d genes" % N_GENES)
    i = 0
    w1 = g[i : i + N_FEATURES * HIDDEN].reshape(HIDDEN, N_FEATURES); i += N_FEATURES * HIDDEN
    b1 = g[i : i + HIDDEN]; i += HIDDEN
    w2 = g[i : i + HIDDEN * N_OUT].reshape(N_OUT, HIDDEN); i += HIDDEN * N_OUT
    b2 = g[i : i + N_OUT]; i += N_OUT
    c, onset = structure(g)
    return w1, b1, w2, b2, c, onset


def head(genes_unpacked, x):
    w1, b1, w2, b2, _, _ = genes_unpacked
    h = np.tanh(w1 @ x + b1)
    return w2 @ h + b2


# ------------------------------------------------------------------ oracles


class PlanarOracle:
    """TwoRoom and PushT: action space is the 2-D motion space; exact/kinematic rollouts."""

    def __init__(self, task):
        self.task = task

    def rollout(self, env, hazard, blocks):
        if self.task == "tworoom":
            start = np.asarray(env.unwrapped.agent_position.numpy(), dtype=np.float64)
            return v4hook.tworoom_rollout(env, start, blocks, hazard)
        u = env.unwrapped
        return v4hook.pusht_rollout(
            env, np.asarray(u.agent.position), np.asarray(u.agent.velocity), blocks, hazard
        )

    def position(self, env):
        if self.task == "tworoom":
            return np.asarray(env.unwrapped.agent_position.numpy(), dtype=np.float64)[:2]
        return np.asarray(env.unwrapped.agent.position, dtype=np.float64)[:2]

    def directions(self, env, hazard, block):
        """Unit action-space directions: away from the hazard, and perpendicular."""
        pos = self.position(env)
        centre = np.asarray(hazard["center"], dtype=np.float64)[:2]
        radial = pos - centre
        norm = np.linalg.norm(radial)
        e = radial / norm if norm > 1e-9 else np.array([1.0, 0.0])
        e_perp = np.array([-e[1], e[0]])
        disp = block.sum(axis=0)
        dn = np.linalg.norm(disp)
        d = disp / dn if dn > 1e-9 else np.zeros(2)
        return e, e_perp, float(d @ e), float(d @ e_perp)


def make_oracle(task):
    if task in ("tworoom", "pusht"):
        return PlanarOracle(task)
    from residual import oracles_extra

    return oracles_extra.make_oracle(task)


# ------------------------------------------------------------------ adapter


class GatedResidual:
    """One genotype, applied block by block. `native_scale` is the measured median |a|."""

    def __init__(self, genes, task, native_scale, *, force_open=False, force_closed=False):
        self.genes = unpack(genes)
        self.task = task
        self.oracle = make_oracle(task)
        self.native_scale = float(native_scale)
        self.c, self.onset = self.genes[4], self.genes[5]
        self.beta_r = self.c * self.native_scale
        self.force_open = force_open
        self.force_closed = force_closed

    def apply(self, env, hazard, proposed, low, high, fraction):
        """Return (block, record). Block is `proposed` itself when the gate stays closed."""
        record = {"gate": 0, "veto": 0, "clear_native": None, "clear_exec": None, "m": None}
        if hazard is None:
            return proposed, record
        rho = float(hazard["radius"])
        block = np.asarray(proposed, dtype=np.float64)
        clear_n, steps_n, _ = self.oracle.rollout(env, hazard, block[None])
        m = float(clear_n[0]) / rho
        record.update(clear_native=float(clear_n[0]), m=m, safe_steps_native=int(steps_n[0]))
        if self.force_closed or (not self.force_open and m >= self.onset):
            return proposed, record
        record["gate"] = 1
        e, e_perp, d_e, d_perp = self.oracle.directions(env, hazard, block)
        x = np.array(
            [
                np.clip(m, -2.0, 3.0) / 3.0,
                steps_n[0] / 5.0,
                d_e,
                d_perp,
                float(fraction),
                float(np.abs(block).mean() / max(self.native_scale, 1e-6)),
                1.0,
            ]
        )
        v, p, l = head(self.genes, x)
        gain = 1.0 + GAIN_BOUND * np.tanh(v)
        residual = self.beta_r * (np.tanh(p) * e + np.tanh(l) * e_perp)
        corrected = np.clip(block * gain + residual[None, :], low, high)
        clear_c, steps_c, _ = self.oracle.rollout(env, hazard, corrected[None])
        record.update(
            gain=float(gain), residual=residual.tolist(), clear_corrected=float(clear_c[0]),
            safe_steps_corrected=int(steps_c[0]),
        )
        if clear_c[0] < clear_n[0]:
            record["veto"] = 1
            record["clear_exec"] = float(clear_n[0])
            return proposed, record
        record["clear_exec"] = float(clear_c[0])
        return corrected.astype(np.asarray(proposed).dtype), record


def random_genotype(rng, sigma=0.5):
    """Random draw from the search's initial distribution; structure genes at their defaults."""
    g = rng.normal(0.0, sigma, N_GENES)
    g[N_WEIGHTS:] = 0.0
    return g
