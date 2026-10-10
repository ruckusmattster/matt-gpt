# Project history

## v1 — notebook + Space (2024)

- Built a character-level GPT in a Kaggle notebook following Karpathy's
  walkthrough; trained for 18k iterations on a Kaggle GPU.
- Exported the Discord messages to `messages.txt` and published them as the
  Kaggle dataset `long-discord`.
- Published the trained checkpoint as the Kaggle model `longtext` (Sept 2024)
  and deployed it on a Gradio Hugging Face Space with top-k / top-p controls.
- The repo held only `messages.txt` and a README linking to the other pieces.

## v2 — this repo (Oct 2026)

Pulled everything into one place and cleaned it up.

**Structure.** One `mattgpt` package shared by training, CLI generation and the
web app, replacing three copies of the same model code with module-level globals.

**Checkpoint compatibility.** Parameter and module names are unchanged
(including the `FeedFoward` spelling and per-head `tril` buffers), so the
released `weights.pth` loads with `strict=True`. Its vocabulary is kept as
`LEGACY_VOCAB`. New checkpoints also store `config`, `vocab` and `iter`.

**Bugs fixed from the Space's `app.py`:**

| Bug | Effect | Fix |
|---|---|---|
| top-p filter indexed `logits[token_ids]` along the batch dimension | `IndexError` whenever top-p < 1 | scatter the mask back to vocab order |
| `encode()` used `dict[c]` | any character outside the vocab (most emoji) crashed generation | unknown characters are dropped |
| `gr.Number` returns a float | `range(400.0)` → `TypeError` unless the UI coerced it | `precision=0` + `int()` |
| empty prompt | zero-length context → indexing error | falls back to a newline |
| position embedding used a global `device` | broke if the model and global disagreed | uses `index.device` |
| unused `tiktoken`, `sentencepiece`, `transformers` deps | slow Space builds | removed |

**Training script.** CLI flags for every hyperparameter, warmup + cosine LR
decay (pass `--min-lr-ratio 1` for the original constant LR), gradient
clipping, best-val checkpointing, resume support, periodic samples.

**Data privacy.** The original `messages.txt` contained a home address, phone
number, personal and third-party email addresses, a password, and ~3,000
Discord user IDs. `scripts/scrub_data.py` removes these; `data/messages.txt` is
now the scrubbed version. The raw export lives only in git-ignored `data/raw/`.

**Finding.** The deployed checkpoint's 212-character vocabulary doesn't match
the Discord data's 534 characters, so it was trained on a different (story)
corpus.

## v2.1 — trained on the Discord data (Oct 2026)

- Trained the default config on the scrubbed `data/messages.txt` on a Kaggle
  GPU (18,000-step schedule, ~0.34 s per step). Validation loss fell fastest in the first
  1,000 steps and bottomed out at **1.44 nats/char at step 9,000**; after that
  the model overfit, and `train.py` kept the step-9,000 weights.
- Samples now sound like the source: 3D-printing, flashlight and Stable
  Diffusion vocabulary, short lowercase lines, without coherence beyond about
  a sentence.
- Added `scripts/export_inference.py` (92 MB training checkpoint → 15 MB fp16
  inference file) and committed the result as `weights/mattgpt.pth`, so the
  repo works straight after cloning.
- Moved the app from Gradio 4.44 to 5.50: 4.44 crashes with current
  FastAPI/Starlette ("unhashable type: 'dict'") and needs `audioop`, which
  Python 3.13 removed.
- Deployment is now a GitHub Action that uploads a clean snapshot (app,
  package, weights) to the Space on each merge, replacing the hand-edited
  Space files and the 88 MB story-model `weights.pth`. Tests run in CI.
- Replaced a real Discord user ID that had been used as test data.
