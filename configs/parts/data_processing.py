from dataclasses import dataclass


@dataclass(frozen=True)
class DataProcessingConfig:
    """Config for processing data.
    Max lengths don't include <SOS> and <EOS> tokens.
    None stands for unlimited size.
    """

    max_path_length: int
    max_name_parts: int
    max_target_parts: int
    wrap_path: bool
    wrap_name: bool
    wrap_target: bool
    split_target: bool
    split_names: bool
    subtoken_vocab_max_size: int = None
    target_vocab_max_size: int = None
    buffer_size: int = 10_000
