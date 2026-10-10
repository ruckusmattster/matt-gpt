# matt-gpt

A small character-level GPT, built from scratch in PyTorch and trained on ~74k of
my own Discord messages, so it talks like me — mostly about 3D printers,
flashlights and Stable Diffusion, and making about as much sense as you'd expect
from 7.5M parameters.

**Try it:** [huggingface.co/spaces/mattyhew/mattgpt](https://huggingface.co/spaces/mattyhew/mattgpt)

```
> anyone know a good upscaler
ah cool, ill see if that doesnt have all the convoy probe
ah cool, also do you know what i do
mine was a lot of ones scale and rails if you can see a

> what flashlight should i get
what flashlight should i get the screw in the hotend sits in some nichia from
there for the way the way and corexy is great

> lol
Hey @user  it's pretty close to 10000 on one of the rails are always a rackup
I've thought the IR can be out but I can't find the time
Everyone is the USA on the game
Bruh I'm the UK
```

It has picked up the vocabulary (hotends, CoreXY, Nichia, Convoy, benchies),
the rhythm (short lowercase lines, "ah cool", "rip", "Bruh") and the topics,
but not much meaning beyond a sentence or so: it only ever sees the last
128 characters, about five messages.

## Results

| | |
|---|---|
| Model | 4 layers × 4 heads, 384-dim, 128-char context, **7.55 M params** |
| Data | 2.77 M characters, 534-character vocabulary, 80/20 contiguous split |
| Training | 18,000-step schedule, batch 128, on a Kaggle GPU (~0.34 s per step) |
| Best checkpoint | step **9,000** (kept automatically; later steps overfit) |
| Validation loss | **1.44 nats/char** (2.08 bits/char, perplexity 4.2) |
| Training loss at best | 1.18 nats/char |

Validation loss during the run:

| step | 0 | 500 | 1,000 | 2,000 | 3,000 | 3,500 | 9,000 (best) |
|---|---|---|---|---|---|---|---|
| train | 6.32 | 1.72 | 1.55 | 1.43 | 1.38 | 1.34 | 1.18 |
| val | 6.32 | 1.73 | 1.60 | 1.51 | 1.48 | 1.46 | **1.44** |

The train/val gap opening up past ~3k steps is the model starting to memorise:
2.2 M training characters is small for 7.5 M parameters, and 18k steps is
~130 passes over it. More data would help more than more training.

## Quickstart

```bash
pip install -r requirements.txt

python generate.py --prompt "anyone know a good upscaler" --num-samples 3
python app.py          # local web UI on http://127.0.0.1:7860
```

The released weights (`weights/mattgpt.pth`, 15 MB) are in the repo, so this
works straight after cloning. `--temperature` (default 0.8) and `--top-k`
(default 40) trade coherence for variety.

## Training it yourself

On a free Kaggle GPU notebook (Settings → Accelerator: GPU, Internet: on):

```
!git clone --depth 1 https://github.com/ruckusmattster/matt-gpt
%cd matt-gpt
!python train.py --data data/messages.txt --out /kaggle/working/mattgpt.pth --eval-iters 100 --sample-every 3000
```

`train.py` keeps the checkpoint with the best validation loss and can resume
with `--resume`. Every hyperparameter is a flag; `python train.py -h` lists them.
To publish a new model, shrink the training checkpoint (which includes optimiser
state) to an inference file and commit it:

```bash
python scripts/export_inference.py mattgpt.pth weights/mattgpt.pth   # 92 MB -> 15 MB, fp16
```

To train on your own messages, scrub them first:

```bash
python scripts/scrub_data.py raw.txt data/messages.txt --redact-file private.txt --report
```

## Deployment

Merging to `main` runs [`deploy-space.yml`](.github/workflows/deploy-space.yml),
which builds a clean copy of just the app, package and weights
(`scripts/build_space.py`) and uploads it to the Space. It needs a Hugging Face
write token saved as the `HF_TOKEN` Actions secret; without it the job skips.

## Repo layout

```
mattgpt/            model, char tokenizer, config, checkpoint I/O, weight download
train.py            training (CLI, resumable, cosine LR, best-val checkpointing)
generate.py         sample from the command line
app.py              Gradio UI (what the Space runs)
weights/mattgpt.pth released model, fp16 inference-only
data/messages.txt   scrubbed training data (see data/README.md)
scripts/            scrub_data.py, export_inference.py, build_space.py
docs/               ARCHITECTURE.md (model details), HISTORY.md (how it got here)
tests/              pytest suite; CI runs it on every push and PR
```

## Roadmap

- [x] Train on the Discord data
- [ ] More data: the model is data-limited, not compute-limited
- [ ] Longer context (512+ chars) so it can follow a conversation
- [ ] Byte-level / BPE tokenizer so emoji-heavy text doesn't bloat the vocabulary
- [ ] Pre-norm blocks + fused attention for a deeper, faster model

## Links

| | |
|---|---|
| Live demo | [HF Space `mattyhew/mattgpt`](https://huggingface.co/spaces/mattyhew/mattgpt) |
| First model (story text, 2024) | [Kaggle model `longtext`](https://www.kaggle.com/models/matthewweinberger/longtext) |

## Acknowledgements

Model code follows Andrej Karpathy's
[*Let's build GPT: from scratch, in code, spelled out*](https://www.youtube.com/watch?v=kCc8FmEb1nY).
