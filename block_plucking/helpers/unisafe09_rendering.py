"""Require the tested headless driver path and real camera pixels before inference."""

import os
import subprocess
from pathlib import Path


def configure_renderer(root):
    manifest = Path(root) / "configs/unisafe09/vulkan-icd.json"
    if not manifest.is_file():
        raise ValueError("The pinned EGL Vulkan manifest is missing")
    # Set both names for the system and Isaac Sim's bundled Vulkan loaders.
    os.environ["VK_DRIVER_FILES"] = str(manifest.resolve())
    os.environ["VK_ICD_FILENAMES"] = str(manifest.resolve())


def verify_renderer(root):
    configure_renderer(root)
    result = subprocess.run(
        ["vulkaninfo", "--summary"], capture_output=True, text=True, timeout=30
    )
    if result.returncode or "NVIDIA GeForce RTX 3090" not in result.stdout:
        raise RuntimeError("The pinned Vulkan path did not enumerate the rented RTX 3090")
    return {"stdout": result.stdout, "stderr": result.stderr, "exit_code": result.returncode}


def validate_cameras(cameras):
    import numpy as np

    if cameras.shape != (2, 128, 128, 3) or cameras.dtype != np.uint8:
        raise ValueError("Camera shape or dtype differs from the frozen model input")
    for index, camera in enumerate(cameras):
        if int(camera.max()) == int(camera.min()):
            raise RuntimeError(f"Camera {index} is a constant frame; renderer is not usable")
