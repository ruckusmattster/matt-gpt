"""Decoder-only transformer, character level.

The architecture follows Andrej Karpathy's "Let's build GPT" walkthrough, with one
difference worth knowing: blocks are *post-norm* (LayerNorm applied after each
residual add, as in the original 2017 Transformer) rather than the pre-norm used
by GPT-2. Module and parameter names are kept identical to the original notebook
so the released checkpoint loads with `load_state_dict(..., strict=True)`.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import functional as F

from .config import GPTConfig


class Head(nn.Module):
    """One head of causal self-attention."""

    def __init__(self, cfg: GPTConfig, head_size: int):
        super().__init__()
        self.key = nn.Linear(cfg.n_embd, head_size, bias=False)
        self.query = nn.Linear(cfg.n_embd, head_size, bias=False)
        self.value = nn.Linear(cfg.n_embd, head_size, bias=False)
        # Lower-triangular mask; stored in the state dict of the original checkpoint.
        self.register_buffer("tril", torch.tril(torch.ones(cfg.block_size, cfg.block_size)))
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        k = self.key(x)  # (B, T, hs)
        q = self.query(x)  # (B, T, hs)
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5  # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)  # (B, T, hs)
        return wei @ v  # (B, T, hs)


class MultiHeadAttention(nn.Module):
    """Several attention heads in parallel, concatenated and projected."""

    def __init__(self, cfg: GPTConfig, num_heads: int, head_size: int):
        super().__init__()
        self.heads = nn.ModuleList([Head(cfg, head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, cfg.n_embd)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedFoward(nn.Module):  # (sic) name kept for checkpoint compatibility
    """Position-wise MLP: Linear -> ReLU -> Linear, 4x expansion."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
            nn.ReLU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    """Transformer block: attention (communication) then MLP (computation)."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        head_size = cfg.n_embd // cfg.n_head
        self.sa = MultiHeadAttention(cfg, cfg.n_head, head_size)
        self.ffwd = FeedFoward(cfg)
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.ln2 = nn.LayerNorm(cfg.n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln1(x + self.sa(x))  # post-norm
        x = self.ln2(x + self.ffwd(x))
        return x


class GPTLanguageModel(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        if cfg.n_embd % cfg.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        self.cfg = cfg
        self.token_embedding_table = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.position_embedding_table = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.blocks = nn.Sequential(*[Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, index: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = index.shape
        if T > self.cfg.block_size:
            raise ValueError(f"sequence length {T} exceeds block_size {self.cfg.block_size}")
        tok_emb = self.token_embedding_table(index)  # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=index.device))  # (T, C)
        x = self.blocks(tok_emb + pos_emb)
        logits = self.lm_head(self.ln_f(x))  # (B, T, vocab)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(B * T, -1), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        index: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Autoregressively extend `index` (B, T) by `max_new_tokens` characters."""
        for _ in range(int(max_new_tokens)):
            index_cond = index[:, -self.cfg.block_size :]
            logits, _ = self(index_cond)
            logits = logits[:, -1, :]
            probs = sampling_probs(logits, temperature=temperature, top_k=top_k, top_p=top_p)
            index_next = torch.multinomial(probs, num_samples=1, generator=generator)
            index = torch.cat((index, index_next), dim=1)
        return index


def sampling_probs(
    logits: torch.Tensor, temperature: float = 1.0, top_k: int = 0, top_p: float = 1.0
) -> torch.Tensor:
    """Turn (B, V) logits into a sampling distribution with temperature, top-k and top-p.

    top_k <= 0 disables top-k; top_p >= 1 disables top-p. At least one token always
    survives filtering.
    """
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    logits = logits / temperature

    if top_k and top_k > 0:
        k = min(int(top_k), logits.size(-1))
        kth = torch.topk(logits, k, dim=-1).values[..., -1, None]
        logits = logits.masked_fill(logits < kth, float("-inf"))

    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        cum = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        remove = cum > top_p
        # shift right so the token that crosses the threshold is kept
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        mask = torch.zeros_like(remove).scatter(-1, sorted_idx, remove)
        logits = logits.masked_fill(mask, float("-inf"))

    return F.softmax(logits, dim=-1)
