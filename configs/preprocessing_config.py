from dataclasses import dataclass


@dataclass(frozen=True)
class PreprocessingConfig:
    """Config for preprocessing data. Max lengths don't include <SOS> and <EOS> tokens.
    -1 stands for size."""

    dataset_name: str
    max_path_length: int
    max_name_parts: int
    max_target_parts: int
    wrap_path: bool
    wrap_name: bool
    wrap_target: bool
    split_target: bool
    split_names: bool
    subtoken_vocab_max_size: int = -1
    target_vocab_max_size: int = -1
    buffer_size: int = 10_000
