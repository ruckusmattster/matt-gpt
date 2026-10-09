"""Checkpoint save/load.

Checkpoint format (a dict saved with torch.save):

    {
        "modelState":     model.state_dict(),        # always present
        "optimizerState": optimizer.state_dict(),    # present in training checkpoints
        "config":         GPTConfig.to_dict(),       # new-format only
        "vocab":          list[str],                 # new-format only
        "iter":           int,                       # new-format only
    }

The released `weights.pth` is a *legacy* checkpoint: it has "modelState" (and
optimizer state, which is why it is ~89 MB for a ~7M-parameter model) but no
config or vocab. Those fall back to GPTConfig() defaults and LEGACY_VOCAB.
"""

from __future__ import annotations

from pathlib import Path

import torch

from .config import GPTConfig
from .model import GPTLanguageModel
from .tokenizer import CharTokenizer


def save_checkpoint(path, model: GPTLanguageModel, tokenizer: CharTokenizer,
                    optimizer=None, iteration: int | None = None) -> None:
    ckpt = {
        "modelState": model.state_dict(),
        "config": model.cfg.to_dict(),
        "vocab": tokenizer.chars,
    }
    if optimizer is not None:
        ckpt["optimizerState"] = optimizer.state_dict()
    if iteration is not None:
        ckpt["iter"] = iteration
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, path)


def load_checkpoint(path, device: str = "cpu"):
    """Load any MattGPT checkpoint. Returns (model, tokenizer, raw_checkpoint_dict)."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    if "modelState" not in ckpt:  # bare state_dict
        ckpt = {"modelState": ckpt}

    if "vocab" in ckpt:
        tokenizer = CharTokenizer(ckpt["vocab"])
    else:
        tokenizer = CharTokenizer.legacy()

    cfg = GPTConfig.from_dict(ckpt.get("config", {}))
    cfg.vocab_size = tokenizer.vocab_size

    state = ckpt["modelState"]
    emb = state["token_embedding_table.weight"]
    if emb.shape[0] != tokenizer.vocab_size:
        raise ValueError(
            f"checkpoint has {emb.shape[0]} token embeddings but the vocabulary has "
            f"{tokenizer.vocab_size} characters; wrong vocab for this checkpoint"
        )

    model = GPTLanguageModel(cfg)
    model.load_state_dict(state)
    model.to(device).eval()
    return model, tokenizer, ckpt
