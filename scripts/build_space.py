"""Assemble the files the Hugging Face Space needs into one folder.

    python scripts/build_space.py build/space

Copies only what the app runs on (app.py, the mattgpt package, the published
weights) and writes a Space README (YAML config header) and a CPU-only
requirements.txt. Deliberately does *not* push git history: the Space gets a
clean snapshot. .github/workflows/deploy-space.yml uploads the folder.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GRADIO_VERSION = "5.50.0"

SPACE_README = f"""---
title: mattgpt
emoji: 👁
colorFrom: pink
colorTo: gray
sdk: gradio
sdk_version: {GRADIO_VERSION}
python_version: "3.10"
app_file: app.py
pinned: false
short_description: A tiny GPT trained from scratch on my Discord messages
---

# mattgpt

A 7.5M-parameter character-level GPT trained from scratch on ~74k of my own
Discord messages. Source, data and training code:
https://github.com/ruckusmattster/matt-gpt

This Space is deployed automatically from that repo; edit there, not here.
"""

# CPU-only torch keeps the Space build small (the default PyPI wheel pulls in
# several GB of CUDA libraries the free CPU hardware can't use). Gradio itself
# is installed by Spaces from sdk_version above.
SPACE_REQUIREMENTS = """--extra-index-url https://download.pytorch.org/whl/cpu
torch==2.5.1+cpu
"""


def build(out: Path) -> list[str]:
    if out.exists():
        shutil.rmtree(out)
    (out / "weights").mkdir(parents=True)
    shutil.copy2(ROOT / "app.py", out / "app.py")
    shutil.copytree(ROOT / "mattgpt", out / "mattgpt", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    weights = ROOT / "weights" / "mattgpt.pth"
    if not weights.exists():
        raise FileNotFoundError(f"{weights} is missing; export it with scripts/export_inference.py")
    shutil.copy2(weights, out / "weights" / "mattgpt.pth")
    (out / "README.md").write_text(SPACE_README, encoding="utf-8")
    (out / "requirements.txt").write_text(SPACE_REQUIREMENTS, encoding="utf-8")
    return sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out", nargs="?", default="build/space")
    args = p.parse_args()
    for f in build(Path(args.out)):
        print(f)


if __name__ == "__main__":
    main()
