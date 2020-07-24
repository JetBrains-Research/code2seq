from dataclasses import dataclass


@dataclass(frozen=True)
class ClassifierConfig:
    classifier_input_size: int
    n_hidden_layers: int
    activation: str
    hidden_size: int
