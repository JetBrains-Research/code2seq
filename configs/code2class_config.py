from dataclasses import dataclass


@dataclass(frozen=True)
class ClassifierConfig:
    hidden_size: int
    classifier_input_size: int
