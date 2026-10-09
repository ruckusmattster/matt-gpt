# Architecture

MattGPT is a decoder-only transformer that predicts the next **character**. It is
built from scratch in PyTorch following Andrej Karpathy's
[*Let's build GPT*](https://www.youtube.com/watch?v=kCc8FmEb1nY) walkthrough.

```
 text ──► char ids (B,T)
            │
   token embedding (V×384) + position embedding (128×384)
            │
   ┌────────▼─────────┐
   │ Block × 4        │   x = LN1(x + MultiHeadAttention(x))     4 heads × 96 dims, causal mask
   │  (post-norm)     │   x = LN2(x + MLP(x))                    384 → 1536 → 384, ReLU
   └────────┬─────────┘
            │
        LayerNorm ──► Linear (384×V) ──► logits over V characters
```

## Hyperparameters (released checkpoint)

| | value |
|---|---|
| Vocabulary `V` | 212 characters (legacy checkpoint) / 534 (Discord data) |
| Context length | 128 characters |
| Embedding dim | 384 |
| Layers × heads | 4 × 4 (head size 96) |
| MLP expansion | 4× (1536) |
| Dropout | 0.2 |
| Parameters | 7.31 M (V=212) / 7.55 M (V=534) |
| Optimiser | Adam-family (AdamW in `train.py`), lr 1e-3, batch 128 |
| Iterations | 18,000 (≈ 295 M characters seen) |

## Things worth knowing

**Character-level.** Every distinct character is a token, so the vocabulary is
whatever characters appear in the corpus. That makes the model tiny and the
tokenizer trivial, but a checkpoint is tied to its exact vocabulary: feed it
text tokenized with a different character set and the ids mean different
letters. New checkpoints store `vocab` inside the file for this reason.

**Post-norm blocks.** LayerNorm is applied *after* each residual add
(`LN(x + f(x))`), as in the original 2017 Transformer, rather than GPT-2's
pre-norm (`x + f(LN(x))`). At 4 layers it trains fine; for a much deeper model
pre-norm is more stable. Kept as-is so the released weights still load.

**Separate attention heads.** Each head is its own `nn.Module` with its own
key/query/value `Linear`s and a stored causal-mask buffer (`tril`). This is
slower than a fused `scaled_dot_product_attention` but readable, and the
parameter names match the checkpoint.

**Sampling.** `generate()` supports temperature, top-k and top-p (nucleus)
sampling. The top-p implementation in the original Space indexed the logits
tensor by token id along the batch dimension, which crashed whenever top-p < 1;
it's fixed here (filter mask is scattered back into vocabulary order) and
covered by a test.

**Checkpoint size.** `weights.pth` is 88.8 MB for 7.3 M parameters because it
also contains the AdamW optimiser state (two moment tensors per parameter):
7.3 M × 4 bytes × 3 ≈ 88 MB. An inference-only export would be ~29 MB.
