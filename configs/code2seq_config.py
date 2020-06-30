from dataclasses import dataclass


@dataclass(frozen=True)
class DecoderConfig:
    decoder_size: int
    num_decoder_layers: int
    beam_width: int = 0


@dataclass(frozen=True)
class EncoderConfig:
    embedding_size: int
    rnn_size: int
    use_bi_rnn: bool
    embedding_dropout: float = 0.0
    rnn_dropout: float = 0.0


@dataclass(frozen=True)
class Code2SeqConfig:
    train_data_path: str
    val_data_path: str
    test_data_path: str

    encoder: EncoderConfig
    decoder: DecoderConfig

    batch_size: int
    val_batch_size: int
    learning_rate: float
    shuffle_data: bool = True
