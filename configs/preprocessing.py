from dataclasses import dataclass


@dataclass(frozen=True)
class PreprocessingConfig:
    """Config for preprocessing data. Max lengths don't include <SOS> and <EOS> tokens.
    -1 stands for size."""

    data_path: str
    max_path_length: int
    max_name_parts: int
    max_target_parts: int
    subtoken_vocab_max_size: int = -1
    target_vocab_max_size: int = -1
