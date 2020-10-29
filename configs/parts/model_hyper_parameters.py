from dataclasses import dataclass


@dataclass(frozen=True)
class ModelHyperParameters:
    n_epochs: int
    patience: int  # early stopping
    batch_size: int
    test_batch_size: int
    learning_rate: float
    weight_decay: float
    decay_gamma: float
    clip_norm: float

    max_context: int
    random_context: bool
    shuffle_data: bool

    save_every_epoch: int = 1
    val_every_epoch: int = 1
    log_every_epoch: int = 10

    optimizer: str = "Momentum"
    nesterov: bool = True
