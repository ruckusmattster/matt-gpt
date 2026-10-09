"""Character-level tokenizer.

Every distinct character in the training text becomes one token. The vocabulary
is the sorted set of characters, so it is fully determined by the corpus: a model
can only be used with the exact vocabulary it was trained with. New checkpoints
store their vocabulary inside the checkpoint; the original `weights.pth` did not,
so its 212-character vocabulary is preserved here as LEGACY_VOCAB.
"""

from __future__ import annotations

# Vocabulary of the released `weights.pth` / Kaggle `longtext.pth` checkpoint,
# copied verbatim from the original app.py. Order matters: index == token id.
LEGACY_VOCAB = [
    '\t', '\n', ' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '<', '=', '>', '?', '@',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
    'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', '\\', ']', '^', '_', '`',
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r',
    's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '}', '~',
    '\x81', '\x8d', '\x8f', '\x90', '\x92', '\x93', '\x94', '\x9d', '\xa0', '¡', '¢', '£', '¤',
    '¥', '¦', '§', '¨', '©', 'ª', '«', '¬', '\xad', '®', '¯', '°', '±', '²', '³', '´', 'µ', '¶',
    '·', '¸', '¹', 'º', '»', '¼', '½', '¾', '¿', 'Â', 'Ã', 'Æ', 'Ç', 'É', 'Ê', 'Ë', 'Ð', 'Ò', '×',
    'Ø', 'Ù', 'à', 'á', 'â', 'ã', 'ä', 'å', 'é', 'í', 'ï', 'ð', 'ñ', 'ó', 'ö', 'ā', 'Œ', 'œ',
    'Š', 'š', 'Ÿ', 'Ž', 'ž', 'ƒ', 'ˆ', '˜', 'і', ' ', ' ', ' ', '​', '‎',
    '–', '—', '―', '‘', '’', '‚', '“', '”', '„', '†', '‡', '•', '…', ' ', ' ', '‪',
    '‰', '′', '‹', '›', '€', '™', '−', '─', '」', 'ﬁ', '﻿', '�', '𝑐', '🌴', '🌹', '🍌', '🙂',
]


class CharTokenizer:
    def __init__(self, chars: list[str]):
        if len(set(chars)) != len(chars):
            raise ValueError("vocabulary contains duplicate characters")
        self.chars = list(chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        return cls(sorted(set(text)))

    @classmethod
    def legacy(cls) -> "CharTokenizer":
        return cls(LEGACY_VOCAB)

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, s: str, strict: bool = False) -> list[int]:
        """Text -> token ids.

        With strict=False (the default) characters outside the vocabulary are
        dropped instead of raising, so a prompt containing an emoji the model
        never saw doesn't crash generation.
        """
        if strict:
            return [self.stoi[c] for c in s]
        return [self.stoi[c] for c in s if c in self.stoi]

    def unknown_chars(self, s: str) -> set[str]:
        return {c for c in s if c not in self.stoi}

    def decode(self, ids) -> str:
        return "".join(self.itos[int(i)] for i in ids)
