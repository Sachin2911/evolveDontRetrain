"""Small shared primitives. No simulator, ML, or cloud SDK imports."""

from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class ComputeError(RuntimeError):
    pass


STORAGE_REPOSITORIES = {
    "baseline": "Sachioster/LeWMExp1_Baseline_Reproduction",
    "ap": "Sachioster/LeWMExp1_AP_Reproduction",
    "adapters": "Sachioster/LeWMExp1_Adapter_Development",
}


def validate_storage(storage):
    """Accept only declared private experiment destinations and fixed layout."""
    if not isinstance(storage, dict) or (
        storage.get("repo_id") not in STORAGE_REPOSITORIES.values()
        or storage.get("repo_type") != "dataset"
        or storage.get("private") is not True
        or storage.get("branch") != "main"
        or storage.get("path_prefix") != "runs"
        or storage.get("remote_runs_root") != "/workspace/LeWM_Experiment1/runs"
        or storage.get("token_env") != "HF_TOKEN"
    ):
        raise ComputeError("Unexpected Experiment 1 private storage settings")
    return dict(storage)


class ApiError(ComputeError):
    def __init__(self, status=None):
        self.status = status
        super().__init__(f"API HTTP {status}" if status else "API connection failed or timed out")


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return default


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".write-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


@contextmanager
def locked(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ComputeError("Another operation holds this lock; check status before retrying.")
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def positive(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("Expected a finite positive number")
    return value


def nonnegative(value):
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError("Expected a finite nonnegative number")
    return value


def valid_id(value):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", str(value)):
        raise ComputeError("Invalid run/session identifier")
    return str(value)


def relative_path(value):
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or ".." in path.parts
        or "\\" in value
        or any(ord(char) < 32 for char in value)
    ):
        raise ComputeError("Expected a safe relative path")
    if str(path) != value or any(part.startswith(".") for part in path.parts):
        raise ComputeError("Hidden, empty, or noncanonical artifact path")
    return path


def safe_path(root, relative):
    root = Path(root).resolve()
    relative_path(relative)
    path = root / relative
    for parent in (path, *path.parents):
        if parent == root:
            break
        if parent.is_symlink():
            raise ComputeError("Symlinks are not allowed in run artifacts")
    if not path.resolve().is_relative_to(root):
        raise ComputeError("Path escapes its run directory")
    return path


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def redact(message, secrets=()):
    result = str(message)
    for secret in secrets:
        if secret:
            result = result.replace(secret, "[redacted]")
    return re.sub(r"\b(?:hf_|gh[pousr]_)[A-Za-z0-9_]{12,}\b", "[redacted]", result)


def reject_secrets(data, secrets):
    if any(secret and secret.encode() in data for secret in secrets):
        raise ComputeError("A credential was found in content intended for transfer or recording")
    if re.search(rb"\b(?:hf_|gh[pousr]_)[A-Za-z0-9_]{20,}\b", data):
        raise ComputeError("Token-shaped content cannot be recorded or deployed")


def load_secrets(project):
    names = ("VAST_API_KEY", "HF_TOKEN", "GHCR_READ_TOKEN", "GITHUB_TOKEN", "WANDB_API_KEY")
    values = {name: os.environ.get(name, "").strip() for name in names}
    path = Path(project) / ".env"
    if path.is_file():
        for line in path.read_text().splitlines():
            name, separator, value = line.removeprefix("export ").partition("=")
            name = name.strip()
            if separator and name in names and not values[name]:
                parts = shlex.split(value, comments=True)
                values[name] = parts[0] if parts else ""
    if not values["VAST_API_KEY"]:
        config = Path(os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config")))
        for path in (config / "vastai/vast_api_key", Path.home() / ".vast_api_key"):
            if path.is_file():
                values["VAST_API_KEY"] = path.read_text().strip()
                break
    return values


def require_secret(secrets, name):
    value = secrets.get(name)
    if not value or any(char.isspace() for char in value):
        raise ComputeError(f"Set {name} in the ignored root .env or environment first")
    return value


class JsonApi:
    def __init__(self, base, token):
        self.base, self.token = base.rstrip("/"), token

    def call(self, path, *, method="GET", payload=None, retry=False):
        for attempt in range(3 if retry or method == "GET" else 1):
            request = Request(
                self.base + path,
                method=method,
                data=None if payload is None else json.dumps(payload, allow_nan=False).encode(),
                headers={
                    "Authorization": f"Bearer {self.token}",
                    "Content-Type": "application/json",
                    "User-Agent": "lewm-compute/1",
                },
            )
            try:
                with urlopen(request, timeout=30) as response:
                    return json.load(response)
            except HTTPError as error:
                failure = ApiError(error.code)
                if error.code not in (429, 500, 502, 503, 504):
                    raise failure from None
            except (URLError, TimeoutError, OSError, ValueError):
                failure = ApiError()
            if attempt == (2 if retry or method == "GET" else 0):
                raise failure from None
            time.sleep(0.5 * (attempt + 1))


def process_identity(pid):
    try:
        text = Path(f"/proc/{int(pid)}/stat").read_text().rsplit(")", 1)[1].split()
        if text[0] == "Z":
            return None
        return {
            "pid": int(pid),
            "start": text[19],
            "boot": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        }
    except (OSError, ValueError, IndexError):
        return None


def process_alive(identity):
    return bool(identity) and process_identity(identity["pid"]) == identity


def error_summary(error):
    if isinstance(error, ComputeError):
        return redact(str(error))
    status = getattr(getattr(error, "response", None), "status_code", None)
    return type(error).__name__ + (f" (HTTP {status})" if status else "")
