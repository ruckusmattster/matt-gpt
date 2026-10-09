"""Train MattGPT on a plain-text corpus.

    python train.py --data data/messages.txt --out checkpoints/mattgpt.pth

Defaults reproduce the original Kaggle run (4 layers, 4 heads, 384-dim, 128-char
context, batch 128, lr 1e-3, 18k iterations, 80/20 split). On a Kaggle P100 or T4
that takes roughly an hour; on CPU, use --max-iters and --batch-size to shrink it.

Resume an interrupted run with --resume checkpoints/mattgpt.pth.
"""

from __future__ import annotations

import argparse
import math
import time

import torch

from mattgpt import CharTokenizer, GPTConfig, GPTLanguageModel, save_checkpoint
from mattgpt.checkpoint import load_checkpoint


def parse_args():
    d = GPTConfig()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data/messages.txt", help="UTF-8 text file to train on")
    p.add_argument("--out", default="checkpoints/mattgpt.pth")
    p.add_argument("--resume", help="checkpoint to continue from (keeps its vocab and config)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=1337)
    for name in ["block_size", "n_embd", "n_head", "n_layer", "batch_size", "max_iters",
                 "eval_interval", "eval_iters"]:
        p.add_argument("--" + name.replace("_", "-"), type=int, default=getattr(d, name))
    for name in ["dropout", "learning_rate", "train_split"]:
        p.add_argument("--" + name.replace("_", "-"), type=float, default=getattr(d, name))
    p.add_argument("--min-lr-ratio", type=float, default=0.1,
                   help="cosine-decay the LR down to this fraction of --learning-rate (1.0 = constant LR, as in the original run)")
    p.add_argument("--warmup-iters", type=int, default=200)
    p.add_argument("--grad-clip", type=float, default=1.0, help="0 disables clipping")
    p.add_argument("--sample-every", type=int, default=0, help="print a sample every N iters (0 = only at the end)")
    return p.parse_args()


def get_lr(it: int, args) -> float:
    if it < args.warmup_iters:
        return args.learning_rate * (it + 1) / args.warmup_iters
    progress = (it - args.warmup_iters) / max(1, args.max_iters - args.warmup_iters)
    min_lr = args.learning_rate * args.min_lr_ratio
    return min_lr + 0.5 * (args.learning_rate - min_lr) * (1 + math.cos(math.pi * min(1.0, progress)))


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    with open(args.data, "r", encoding="utf-8") as f:
        text = f.read()

    start_iter = 0
    if args.resume:
        model, tok, ckpt = load_checkpoint(args.resume, device=args.device)
        cfg = model.cfg
        start_iter = ckpt.get("iter", 0)
        unknown = tok.unknown_chars(text)
        if unknown:
            print(f"warning: {len(unknown)} characters in the data are not in the checkpoint vocab; they will be dropped")
    else:
        tok = CharTokenizer.from_text(text)
        cfg = GPTConfig(
            vocab_size=tok.vocab_size, block_size=args.block_size, n_embd=args.n_embd,
            n_head=args.n_head, n_layer=args.n_layer, dropout=args.dropout,
            batch_size=args.batch_size, max_iters=args.max_iters, learning_rate=args.learning_rate,
            eval_interval=args.eval_interval, eval_iters=args.eval_iters, train_split=args.train_split,
        )
        model = GPTLanguageModel(cfg).to(args.device)
        ckpt = {}

    data = torch.tensor(tok.encode(text), dtype=torch.long)
    n = int(args.train_split * len(data))
    splits = {"train": data[:n], "val": data[n:]}
    print(f"corpus: {len(text):,} chars | vocab: {tok.vocab_size} | train {n:,} / val {len(data) - n:,} tokens")
    print(f"model: {model.num_params() / 1e6:.2f}M parameters on {args.device}")

    def get_batch(split):
        d = splits[split]
        ix = torch.randint(len(d) - cfg.block_size - 1, (args.batch_size,))
        x = torch.stack([d[i : i + cfg.block_size] for i in ix])
        y = torch.stack([d[i + 1 : i + cfg.block_size + 1] for i in ix])
        return x.to(args.device), y.to(args.device)

    @torch.no_grad()
    def estimate_loss():
        model.eval()
        out = {}
        for split in ("train", "val"):
            losses = torch.zeros(args.eval_iters)
            for k in range(args.eval_iters):
                _, loss = model(*get_batch(split))
                losses[k] = loss.item()
            out[split] = losses.mean().item()
        model.train()
        return out

    def sample(n_chars=300):
        model.eval()
        ctx = torch.tensor([tok.encode("\n")], dtype=torch.long, device=args.device)
        out = tok.decode(model.generate(ctx, n_chars, top_k=50)[0].tolist())
        model.train()
        return out

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    if "optimizerState" in ckpt:
        optimizer.load_state_dict(ckpt["optimizerState"])

    best_val = float("inf")
    t0 = time.time()
    model.train()
    for it in range(start_iter, args.max_iters):
        lr = get_lr(it, args)
        for g in optimizer.param_groups:
            g["lr"] = lr

        if it % args.eval_interval == 0 or it == args.max_iters - 1:
            losses = estimate_loss()
            print(f"iter {it:6d} | train {losses['train']:.4f} | val {losses['val']:.4f} | lr {lr:.2e} | {time.time() - t0:.0f}s")
            if losses["val"] < best_val:
                best_val = losses["val"]
                save_checkpoint(args.out, model, tok, optimizer, iteration=it)

        if args.sample_every and it and it % args.sample_every == 0:
            print("-" * 40 + "\n" + sample() + "\n" + "-" * 40)

        _, loss = model(*get_batch("train"))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

    print(f"done. best val loss {best_val:.4f}; best checkpoint saved to {args.out}")
    print("sample:\n" + sample())


if __name__ == "__main__":
    main()
