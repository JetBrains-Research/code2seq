from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class ClassifierConfig:
    classifier_input_size: int
    n_hidden_layers: int
    activation: torch.nn.functional
    hidden_size: int
