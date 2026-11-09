"""Small deterministic scorers, with one causal memory update per decision.

The caller supplies only observations already available at that decision. Each
candidate sees the same history. No simulator or model is imported here.
"""

from collections import deque

import numpy as np

FEATURE_DIM = 20
STREAM_DIM = 8
HIDDEN = 16
HISTORY = 4
ARCHITECTURES = ("mlp", "history", "gru")


def parameter_shapes(architecture):
    if architecture not in ARCHITECTURES:
        raise ValueError("Unknown architecture: " + architecture)
    extra = {"mlp": 0, "history": HISTORY * STREAM_DIM, "gru": HIDDEN}[architecture]
    shapes = []
    if architecture == "gru":
        shapes.extend(
            [(3 * HIDDEN, STREAM_DIM), (3 * HIDDEN, HIDDEN), (3 * HIDDEN,), (3 * HIDDEN,)]
        )
    return shapes + [(HIDDEN, FEATURE_DIM + extra), (HIDDEN,), (HIDDEN,)]


def parameter_count(architecture):
    return sum(int(np.prod(shape)) for shape in parameter_shapes(architecture))


def initial_parameters(architecture, seed):
    rng = np.random.default_rng(seed)
    arrays = []
    for shape in parameter_shapes(architecture):
        arrays.append(
            rng.normal(0, 1 / np.sqrt(shape[1]), shape) if len(shape) == 2 else np.zeros(shape)
        )
    return np.concatenate([a.reshape(-1) for a in arrays])


class Scorer:
    def __init__(self, architecture, parameters):
        self.architecture = architecture
        flat = np.asarray(parameters, dtype=np.float64)
        if flat.shape != (parameter_count(architecture),) or not np.isfinite(flat).all():
            raise ValueError("Invalid scorer parameters")
        self.parameters = flat.copy()
        self.arrays, offset = [], 0
        for shape in parameter_shapes(architecture):
            size = int(np.prod(shape))
            self.arrays.append(flat[offset : offset + size].reshape(shape).copy())
            offset += size
        # This explicit bypass also covers a zero head with nonzero memory weights.
        self.neutral = bool(np.count_nonzero(self.arrays[-1]) == 0)
        self.reset()

    def reset(self):
        self.history = deque([np.zeros(STREAM_DIM) for _ in range(HISTORY)], maxlen=HISTORY)
        self.hidden = np.zeros(HIDDEN)
        self.decision = -1

    def observe(self, stream, decision):
        if decision != self.decision + 1:
            raise ValueError("Memory must update exactly once in consecutive decision order")
        stream = np.asarray(stream, dtype=np.float64)
        if stream.shape != (STREAM_DIM,) or not np.isfinite(stream).all():
            raise ValueError("Invalid causal memory stream")
        self.decision = decision
        self.history.append(stream.copy())
        if self.architecture == "gru":
            wi, wh, bi, bh = self.arrays[:4]
            ir, iz, inn = np.split(wi @ stream + bi, 3)
            hr, hz, hn = np.split(wh @ self.hidden + bh, 3)
            reset = 1 / (1 + np.exp(-np.clip(ir + hr, -60, 60)))
            update = 1 / (1 + np.exp(-np.clip(iz + hz, -60, 60)))
            candidate = np.tanh(inn + reset * hn)
            self.hidden = (1 - update) * candidate + update * self.hidden

    def scores(self, features):
        features = np.asarray(features, dtype=np.float64)
        if features.shape != (8, FEATURE_DIM) or not np.isfinite(features).all():
            raise ValueError("Expected eight finite twenty-field candidates")
        if self.decision < 0:
            raise ValueError("Observe the current causal stream before scoring")
        extra = (
            np.concatenate(self.history)
            if self.architecture == "history"
            else self.hidden
            if self.architecture == "gru"
            else np.empty(0)
        )
        inputs = np.concatenate([features, np.broadcast_to(extra, (8, len(extra)))], axis=1)
        weight, bias, head = self.arrays[-3:]
        return -features[:, 1] + np.tanh(np.tanh(inputs @ weight.T + bias) @ head)

    def choose(self, features):
        return int(np.argmax(self.scores(features)))
