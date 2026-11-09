"""Resolve relocated local assets without changing frozen protocol JSON."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SAFEBLOCK = ROOT / "SafeBlock"


def project_path(value):
    """Map historical repository asset/output paths into the SafeBlock workspace."""
    path = Path(value)
    if path.is_absolute():
        if not path.is_relative_to(ROOT):
            return path
        path = path.relative_to(ROOT)
    for prefix in ("data/unisafe09", "data/safeblock", "configs/unisafe09", "outputs"):
        if path.is_relative_to(prefix):
            return SAFEBLOCK / path
    return ROOT / path
