from dataclasses import dataclass

from .model_hyperparameters_config import ModelHyperparameters
from .modules_config import EncoderConfig, DecoderConfig


@dataclass(frozen=True)
class Code2SeqConfig:
    encoder_config: EncoderConfig
    decoder_config: DecoderConfig
    hyperparams: ModelHyperparameters
