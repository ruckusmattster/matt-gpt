"""Shrink a training checkpoint to an inference-only file.

    python scripts/export_inference.py checkpoints/mattgpt.pth weights/mattgpt.pth

A training checkpoint holds the AdamW optimiser state (two extra tensors per
parameter), so it is ~3x larger than the model itself. This keeps only the model
weights, config and vocab, and by default stores the weights as float16, halving
them again (7.55M params -> ~15 MB instead of ~91 MB). load_checkpoint() casts
them back to float32 when loading, so nothing else needs to change.

The causal-mask buffers (`tril`) are dropped too: they're all ones-and-zeros and
are rebuilt by the model constructor.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mattgpt.checkpoint import load_checkpoint  # noqa: E402


def export(src, dst, dtype: torch.dtype = torch.float16) -> dict:
    model, tok, raw = load_checkpoint(src)
    state = {
        k: (v.to(dtype) if v.is_floating_point() else v)
        for k, v in model.state_dict().items()
        if not k.endswith(".tril")
    }
    out = {"modelState": state, "config": model.cfg.to_dict(), "vocab": tok.chars}
    if "iter" in raw:
        out["iter"] = raw["iter"]
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, dst)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("src")
    p.add_argument("dst")
    p.add_argument("--fp32", action="store_true", help="keep float32 weights")
    args = p.parse_args()
    export(args.src, args.dst, torch.float32 if args.fp32 else torch.float16)
    a, b = Path(args.src).stat().st_size, Path(args.dst).stat().st_size
    print(f"{args.src}: {a / 1e6:.1f} MB -> {args.dst}: {b / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
