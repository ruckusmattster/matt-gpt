import torch

from mattgpt import CharTokenizer, GPTConfig, GPTLanguageModel, LEGACY_VOCAB, load_checkpoint, save_checkpoint
from mattgpt.model import sampling_probs


def tiny_cfg(vocab_size=20):
    return GPTConfig(vocab_size=vocab_size, block_size=16, n_embd=32, n_head=4, n_layer=2, dropout=0.0)


def test_legacy_vocab_matches_released_checkpoint():
    # The released weights.pth has a 212-row token embedding.
    assert len(LEGACY_VOCAB) == 212
    assert len(set(LEGACY_VOCAB)) == 212


def test_default_config_param_count_matches_release():
    model = GPTLanguageModel(GPTConfig())
    # ~7.2M params -> ~29 MB fp32, ~88 MB with AdamW moments, matching the 88.8 MB file.
    assert 7.0e6 < model.num_params() < 7.5e6


def test_state_dict_keys_match_original_naming():
    keys = set(GPTLanguageModel(GPTConfig()).state_dict())
    for k in ["token_embedding_table.weight", "position_embedding_table.weight",
              "blocks.0.sa.heads.0.key.weight", "blocks.0.sa.heads.0.tril",
              "blocks.0.sa.proj.weight", "blocks.3.ffwd.net.2.bias",
              "blocks.0.ln1.weight", "ln_f.weight", "lm_head.weight"]:
        assert k in keys, k


def test_tokenizer_roundtrip_and_unknown_chars():
    tok = CharTokenizer.from_text("hello world")
    assert tok.decode(tok.encode("hello")) == "hello"
    assert tok.encode("hi🙂") == tok.encode("h")  # unknown chars dropped, no KeyError
    assert tok.unknown_chars("hi🙂") == {"i", "🙂"}


def test_forward_shapes_and_loss():
    model = GPTLanguageModel(tiny_cfg())
    x = torch.randint(0, 20, (3, 16))
    logits, loss = model(x, x)
    assert logits.shape == (3, 16, 20)
    assert loss.item() > 0


def test_generate_handles_long_context():
    model = GPTLanguageModel(tiny_cfg()).eval()
    ctx = torch.randint(0, 20, (1, 40))  # longer than block_size
    out = model.generate(ctx, 10, top_k=5, top_p=0.9)
    assert out.shape == (1, 50)


def test_top_p_keeps_nucleus_only():
    # probs ~ [0.6, 0.3, 0.1]; top_p=0.7 must keep tokens 0 and 1 only.
    logits = torch.log(torch.tensor([[0.1, 0.6, 0.3]]))
    p = sampling_probs(logits, top_p=0.7)
    assert p[0, 0] == 0
    assert torch.allclose(p[0, 1:].sum(), torch.tensor(1.0))


def test_top_k():
    logits = torch.tensor([[1.0, 5.0, 3.0, 4.0]])
    p = sampling_probs(logits, top_k=2)
    assert (p[0] > 0).tolist() == [False, True, False, True]


def test_checkpoint_roundtrip(tmp_path):
    tok = CharTokenizer.from_text("abcdefghijklmnopqrst")
    model = GPTLanguageModel(tiny_cfg(tok.vocab_size))
    path = tmp_path / "m.pth"
    save_checkpoint(path, model, tok, torch.optim.AdamW(model.parameters()), iteration=7)
    m2, tok2, raw = load_checkpoint(path)
    assert tok2.chars == tok.chars and raw["iter"] == 7
    for a, b in zip(model.state_dict().values(), m2.state_dict().values()):
        assert torch.equal(a, b)


def test_legacy_checkpoint_format_loads(tmp_path):
    # Mimic weights.pth: only modelState (+ optimizerState), no config or vocab.
    model = GPTLanguageModel(GPTConfig())
    path = tmp_path / "legacy.pth"
    torch.save({"modelState": model.state_dict(), "optimizerState": {}}, path)
    m2, tok, _ = load_checkpoint(path)
    assert tok.chars == LEGACY_VOCAB
    assert m2.cfg.n_layer == 4
