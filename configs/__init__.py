from .preprocessing import PreprocessingConfig


def get_preprocessing_config_code2seq_params(data_path: str):
    return PreprocessingConfig(
        data_path=data_path,
        max_path_length=8,
        max_name_parts=5,
        max_target_parts=6,
        subtoken_vocab_max_size=190000,
        target_vocab_max_size=27000,
    )
