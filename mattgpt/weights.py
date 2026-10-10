"""Locate the released weights, downloading them from GitHub if needed."""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = REPO_ROOT / "weights" / "mattgpt.pth"
WEIGHTS_URL = "https://github.com/ruckusmattster/matt-gpt/raw/main/weights/mattgpt.pth"


def default_weights(download: bool = True) -> Path:
    """Path to weights/mattgpt.pth; fetched from GitHub if it isn't on disk.

    The MATTGPT_WEIGHTS environment variable overrides the location.
    """
    path = Path(os.environ.get("MATTGPT_WEIGHTS", DEFAULT_WEIGHTS))
    if path.exists():
        return path
    if not download:
        raise FileNotFoundError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    print(f"downloading weights from {WEIGHTS_URL} ...")
    urllib.request.urlretrieve(WEIGHTS_URL, tmp)
    tmp.replace(path)
    return path
