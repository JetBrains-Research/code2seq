from dataclasses import dataclass
from os.path import join


@dataclass
class PreprocessingConfig:
    """Config for preprocessing data. Max lengths don't include <SOS> and <EOS> tokens.
    -1 stands for size."""

    dataset_name: str
    max_path_length: int
    max_name_parts: int
    max_target_parts: int
    subtoken_vocab_max_size: int = -1
    target_vocab_max_size: int = -1
    buffer_size: int = 10_000
    max_paths_per_label: int = 1000
    data_root: str = "data"
    data_path: str = None

    def __post_init__(self):
        self.data_path = join(self.data_root, self.dataset_name)
