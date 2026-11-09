"""Publish immutable run artifacts for periodic uploads from an experiment process."""

import hashlib
import os
from pathlib import Path
import shutil
import tempfile
import time

from compute_common import ComputeError, atomic_json, safe_path, sha256


def report_progress(completed, total, *, unit="episodes", run_dir=None):
    """Update compact live status; record scientific metrics separately as artifacts."""
    if not isinstance(completed, int) or not isinstance(total, int) or not 0 <= completed <= total:
        raise ValueError("Progress requires integer counts with 0 <= completed <= total")
    root = Path(run_dir or os.environ["LEWM_RUN_DIR"])
    atomic_json(
        root / ".control/progress.json",
        {
            "completed": completed,
            "total": total,
            "unit": str(unit)[:80],
            "epoch": time.time(),
        },
    )


def mark_ready(root, relative):
    path = safe_path(root, relative)
    if not path.is_file():
        raise ComputeError("Only regular, completed files can be marked ready")
    record = {"path": relative, "bytes": path.stat().st_size, "sha256": sha256(path)}
    key = hashlib.sha256(relative.encode()).hexdigest()
    atomic_json(Path(root) / ".control" / "ready" / f"{key}.json", record)
    return record


def publish_artifact(relative, *, data=None, source=None, run_dir=None):
    """Copy bytes or a closed source file, then publish without replacing prior output.

    Examples: publish_artifact('metrics/chunk-0001.jsonl', data=encoded_rows)
              publish_artifact('artifacts/checkpoint-0100.pt', source=closed_checkpoint)
    """
    if (data is None) == (source is None):
        raise ValueError("Provide exactly one of data or source")
    root = Path(run_dir or os.environ["LEWM_RUN_DIR"])
    if relative.split("/", 1)[0] not in ("metrics", "logs", "artifacts"):
        raise ComputeError("Publish under metrics/, logs/, or artifacts/")
    destination = safe_path(root, relative)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".partial-", dir=destination.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            if source is None:
                stream.write(data.encode() if isinstance(data, str) else data)
            else:
                with Path(source).open("rb") as incoming:
                    shutil.copyfileobj(incoming, stream)
            stream.flush()
            os.fsync(stream.fileno())
        # link is atomic and fails if the destination exists; never overwrite a checkpoint.
        os.link(temporary, destination)
        return mark_ready(root, relative)
    finally:
        Path(temporary).unlink(missing_ok=True)
