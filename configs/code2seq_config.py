from dataclasses import dataclass


@dataclass(frozen=True)
class EncoderConfig:
    embedding_size: int
    rnn_size: int
    use_bi_rnn: bool
    embedding_dropout: float
    rnn_num_layers: int
    rnn_dropout: float


@dataclass(frozen=True)
class DecoderConfig:
    decoder_size: int
    embedding_size: int
    num_decoder_layers: int
    rnn_dropout: float
    teacher_forcing: float
    beam_width: int = 0


@dataclass(frozen=True)
class Code2SeqConfig:
    train_data_path: str
    val_data_path: str
    test_data_path: str

    encoder: EncoderConfig
    decoder: DecoderConfig

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
