from dataclasses import asdict, dataclass


@dataclass
class GPTConfig:
    """Model + training hyperparameters.

    Defaults are the exact values used for the released `longtext` / `weights.pth`
    checkpoint, so a default config always loads those weights.
    """

    # model
    vocab_size: int = 212
    block_size: int = 128  # context length in characters
    n_embd: int = 384
    n_head: int = 4
    n_layer: int = 4
    dropout: float = 0.2

    # training
    batch_size: int = 128
    max_iters: int = 18_000
    learning_rate: float = 1e-3
    eval_interval: int = 500
    eval_iters: int = 300
    train_split: float = 0.8

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "GPTConfig":
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)
