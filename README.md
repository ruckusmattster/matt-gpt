# matt-gpt

A small character-level GPT, built from scratch in PyTorch, with the goal of a
model that writes like me — trained on ~74k of my own Discord messages.

**Try it:** [huggingface.co/spaces/mattyhew/mattgpt](https://huggingface.co/spaces/mattyhew/mattgpt)

```
prompt: then a man named marcus walked in

then a man named marcus walked in the park. He had a surprises in comb!
Timmy wondered what was in the dirt and in the water. The water was facient.
One day, they went to the shelter with their silvery food. ...
```

## Status

| Piece | State |
|---|---|
| Model architecture & training code | ✅ done (`mattgpt/`, `train.py`) |
| Story-text checkpoint (`longtext` / `weights.pth`) | ✅ trained, deployed on the Space |
| Discord dataset | ✅ collected and scrubbed of personal info (`data/`) |
| Discord-trained "talks like Matt" checkpoint | ⏳ not trained yet — see [Roadmap](#roadmap) |

The checkpoint currently deployed was trained on a **story corpus, not the Discord
messages**: its 212-character vocabulary doesn't match the 534 characters in
`messages.txt` (no emoji, but Windows-1252 artefacts like `â` and `€`), and it
writes children's-story prose. That's why the Space's tagline is *"i tell
stories"*. Training on `data/messages.txt` is the next step.

## Repo layout

```
mattgpt/
  config.py       GPTConfig dataclass (defaults = the released checkpoint)
  tokenizer.py    CharTokenizer + LEGACY_VOCAB for the released weights
  model.py        transformer + generate() with temperature / top-k / top-p
  checkpoint.py   save/load; reads both legacy and new checkpoint formats
train.py          training script (CLI; resumable; cosine LR, grad clipping)
generate.py       sample from a checkpoint from the command line
app.py            Gradio UI — the Hugging Face Space
scripts/
  scrub_data.py   strip emails / phones / addresses / Discord IDs from raw exports
data/
  messages.txt    scrubbed training data (see data/README.md)
docs/
  ARCHITECTURE.md model details and design notes
  HISTORY.md      how the project got here, and what changed in this rewrite
tests/            pytest suite (model, sampling, checkpoints, scrubber)
```

## Quickstart

```bash
pip install -r requirements.txt

# sample from the released weights (downloads weights.pth from the Space)
python generate.py --download --prompt "once upon a time" --top-k 50 --seed 0

# run the web UI locally
python app.py

# train on the Discord data (defaults = original run; ~1 h on a Kaggle T4/P100)
python train.py --data data/messages.txt --out checkpoints/mattgpt.pth --sample-every 2000

# sample from your own checkpoint
python generate.py --ckpt checkpoints/mattgpt.pth --prompt "anyone know a good upscaler"
```

CPU-sized run to check everything works (~2 min):

```bash
python train.py --n-layer 2 --n-embd 128 --batch-size 32 --max-iters 400 \
    --eval-interval 200 --eval-iters 20 --warmup-iters 50 --out checkpoints/smoke.pth
```

Tests: `pip install pytest && pytest -q`

## Training on Kaggle

Free GPUs are how this was trained originally. In a Kaggle notebook with GPU on:

```python
!git clone https://github.com/ruckusmattster/matt-gpt && cd matt-gpt
!cd matt-gpt && python train.py --data data/messages.txt --out /kaggle/working/mattgpt.pth --sample-every 2000
```

Save `/kaggle/working/mattgpt.pth` as a Kaggle Model or download it from the
notebook's Output tab.

## Deploying to the Hugging Face Space

The Space needs `app.py`, the `mattgpt/` folder, `requirements.txt` and a
checkpoint named `weights.pth` at the top level. New-format checkpoints carry
their own vocabulary and config, so swapping in a Discord-trained checkpoint is
just replacing `weights.pth`; no code change.

## Model

4-layer, 4-head, 384-dim decoder-only transformer, 128-character context,
7.3 M parameters. Full details and design notes in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Roadmap

- [ ] Train on `data/messages.txt` with the default config and publish it as the new `weights.pth`
- [ ] Compare against a fine-tune of the story checkpoint (needs vocabulary remapping, since the character sets differ)
- [ ] Try a byte-level or BPE tokenizer so emoji-heavy text doesn't bloat the vocab
- [ ] Export inference-only weights (~29 MB instead of 89 MB with optimiser state)
- [ ] Longer context (128 chars is ~5 messages)

## Links

| | |
|---|---|
| Live demo | [HF Space `mattyhew/mattgpt`](https://huggingface.co/spaces/mattyhew/mattgpt) |
| Dataset | [Kaggle `long-discord`](https://www.kaggle.com/datasets/matthewweinberger/long-discord) |
| Released weights | [Kaggle model `longtext`](https://www.kaggle.com/models/matthewweinberger/longtext) |
| Original notebook | [Kaggle `notebookf8a50bdd57`](https://www.kaggle.com/code/matthewweinberger/notebookf8a50bdd57) |

## Acknowledgements

Model code follows Andrej Karpathy's
[*Let's build GPT: from scratch, in code, spelled out*](https://www.youtube.com/watch?v=kCc8FmEb1nY).
