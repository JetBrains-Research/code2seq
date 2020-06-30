from .preprocessing_config import PreprocessingConfig
from .code2seq_config import Code2SeqConfig, DecoderConfig


def get_preprocessing_config_code2seq_params(dataset_name: str):
    return PreprocessingConfig(
        dataset_name=dataset_name,
        max_path_length=8,
        max_name_parts=5,
        max_target_parts=6,
        subtoken_vocab_max_size=190000,
        target_vocab_max_size=27000,
    )
