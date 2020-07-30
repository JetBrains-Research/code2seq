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
class ClassifierConfig:
    classifier_input_size: int
    n_hidden_layers: int
    activation: str
    hidden_size: int
