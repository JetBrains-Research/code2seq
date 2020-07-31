from dataclasses import dataclass

from .model_hyperparameters_config import ModelHyperparameters
from .modules_config import EncoderConfig, ClassifierConfig


@dataclass(frozen=True)
class Code2ClassConfig:
    encoder_config: EncoderConfig
    classifier_config: ClassifierConfig
    hyperparams: ModelHyperparameters
