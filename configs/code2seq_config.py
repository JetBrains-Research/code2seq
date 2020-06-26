from dataclasses import dataclass


@dataclass(frozen=True)
class Code2SeqConfig:
    train_data_path: str
    val_data_path: str
    test_data_path: str
    batch_size: int
    val_batch_size: int
    learning_rate: float
    shuffle_data: bool = True
