"""Generate text from a MattGPT checkpoint.

    python generate.py --ckpt weights/weights.pth --prompt "then a man named marcus walked in"
    python generate.py --download --prompt "once upon a time"   # fetch weights from Hugging Face first
"""

from __future__ import annotations

import argparse
import sys

import torch

from mattgpt import load_checkpoint

HF_REPO = "mattyhew/mattgpt"
HF_FILE = "weights.pth"


def download_weights() -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=HF_REPO, repo_type="space", filename=HF_FILE)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default="weights/weights.pth")
    p.add_argument("--download", action="store_true", help=f"download {HF_FILE} from the {HF_REPO} Space")
    p.add_argument("--prompt", default="\n")
    p.add_argument("--max-new-tokens", type=int, default=400)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--num-samples", type=int, default=1)
    p.add_argument("--seed", type=int)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    ckpt_path = download_weights() if args.download else args.ckpt
    model, tok, _ = load_checkpoint(ckpt_path, device=args.device)

    dropped = tok.unknown_chars(args.prompt)
    if dropped:
        print(f"note: dropping characters not in the vocabulary: {''.join(sorted(dropped))!r}", file=sys.stderr)
    ids = tok.encode(args.prompt) or tok.encode("\n")

    gen = torch.Generator(device=args.device)
    if args.seed is not None:
        gen.manual_seed(args.seed)

    for i in range(args.num_samples):
        ctx = torch.tensor([ids], dtype=torch.long, device=args.device)
        out = model.generate(ctx, args.max_new_tokens, temperature=args.temperature,
                             top_k=args.top_k, top_p=args.top_p, generator=gen)
        if args.num_samples > 1:
            print(f"=== sample {i + 1} ===")
        print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
