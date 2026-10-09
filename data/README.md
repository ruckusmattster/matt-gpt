# Dataset: `messages.txt`

My own Discord messages, exported one message per line, oldest-to-newest order
not guaranteed. Also published on Kaggle as
[`matthewweinberger/long-discord`](https://www.kaggle.com/datasets/matthewweinberger/long-discord).

| | |
|---|---|
| Messages (lines) | ~74,300 |
| Characters | ~2.77 M (scrubbed) |
| Distinct characters | ~530 (lots of emoji) |
| Median message length | 26 characters |
| Language | English, very informal: Stable Diffusion help, 3D printing, flashlights, gaming, chatting with friends |

## Scrubbing

This file is the **scrubbed** version, produced with:

```bash
python scripts/scrub_data.py data/raw/messages.raw.txt data/messages.txt \
    --redact-file data/redact.local.txt --report
```

which replaces emails, phone numbers, postcodes, street addresses, Discord user /
role / channel IDs and custom-emoji IDs with plain placeholders (`[email]`,
`@user`, ...), and redacts any exact strings listed in the private
`data/redact.local.txt` (passwords etc.). Both `data/raw/` and `*.local.txt` are
git-ignored.

Placeholders are ASCII, so scrubbing doesn't add new characters to the vocabulary.
The address regex has a few harmless false positives (e.g. "2 hour drive"),
which is an acceptable cost.

## Train/val split

`train.py` encodes the whole file and takes the first 80% of characters for
training and the last 20% for validation — a contiguous split, so validation
loss measures generalisation to later messages rather than to shuffled ones.
