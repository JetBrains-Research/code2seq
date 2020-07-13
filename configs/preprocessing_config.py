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
    data_root: str = "data"
    data_path: str = None
    train_data_path: str = None
    train: float = None
    test: float = None
    val: float = None
    shuffle: bool = True

    def __post_init__(self):
        self.data_path = join(self.data_root, self.dataset_name)
