"""Durable manifests and a study ledger that charges failures and partial panels."""

from contextlib import contextmanager
import base64
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import time
import zlib


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def serial(value):
    if isinstance(value, dict):
        return {str(k): serial(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serial(v) for v in value]
    if (
        getattr(value, "ndim", None) == 3
        and str(getattr(value, "dtype", "")) == "uint8"
        and value.shape[-1] == 3
    ):
        raw = value.tobytes(order="C")
        return {
            "encoding": "rgb-uint8-zlib-base64",
            "shape": list(value.shape),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "data": base64.b64encode(zlib.compress(raw, level=1)).decode("ascii"),
        }
    if hasattr(value, "tolist"):
        return serial(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite": str(value)}
    if isinstance(value, Path):
        return str(value)
    return value


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(serial(value), stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def publish(name, value):
    from run_io import publish_artifact

    if os.environ.get("QD_TASK"):
        category, tail = name.split("/", 1)
        name = category + "/" + os.environ["QD_TASK"] + "/" + tail
    return publish_artifact(name, data=json.dumps(serial(value), allow_nan=False) + "\n")


class Ledger:
    """Count attempted actions; an interrupted reservation stays fully charged.

    One worker owns each task. The lock also prevents accidental duplicate jobs
    on that host. Reservations are released only on a durable, explicit finish.
    No case identity can be reused, including after a failed attempt.
    """

    def __init__(self, path, limits, protocol_sha256):
        self.path = Path(path)
        self.limits = dict(limits)
        self.protocol_sha256 = protocol_sha256

    @contextmanager
    def locked(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.with_suffix(".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            state = (
                read(self.path)
                if self.path.exists()
                else {
                    "protocol_sha256": self.protocol_sha256,
                    "limits": self.limits,
                    "attempts": {},
                }
            )
            if state["protocol_sha256"] != self.protocol_sha256 or state["limits"] != self.limits:
                raise ValueError("Ledger protocol or limits changed")
            yield state
            write(self.path, state)

    @staticmethod
    def charged(state, phase):
        return sum(a["charged"] for a in state["attempts"].values() if a["phase"] == phase)

    def remaining(self, phase):
        with self.locked() as state:
            return self.limits[phase] - self.charged(state, phase)

    def reserve(self, identity, phase, actions):
        if type(actions) is not int or actions < 0 or phase not in self.limits:
            raise ValueError("Invalid action reservation")
        with self.locked() as state:
            if identity in state["attempts"]:
                raise ValueError("Attempt identity already exists: " + identity)
            if self.charged(state, phase) + actions > self.limits[phase]:
                raise ValueError("Phase action budget exhausted: " + phase)
            state["attempts"][identity] = {
                "phase": phase,
                "reserved": actions,
                "charged": actions,
                "status": "reserved",
                "started_epoch": time.time(),
                "run_id": Path(os.environ["LEWM_RUN_DIR"]).name,
            }

    def finish(self, identity, counts, completed):
        attempted = int(counts.get("policy_actions_attempted", 0))
        with self.locked() as state:
            attempt = state["attempts"][identity]
            if attempt["status"] != "reserved" or not 0 <= attempted <= attempt["reserved"]:
                raise ValueError("Invalid attempt completion")
            attempt.update(
                charged=attempted,
                counts=dict(counts),
                status="completed" if completed else "failed",
                ended_epoch=time.time(),
            )
