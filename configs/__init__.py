from os.path import join

from .preprocessing_config import PreprocessingConfig
from .code2seq_config import Code2SeqConfig, DecoderConfig, EncoderConfig


def get_preprocessing_config_code2seq_params(dataset_name: str) -> PreprocessingConfig:
    return PreprocessingConfig(
        dataset_name=dataset_name,
        max_path_length=8,
        max_name_parts=5,
        max_target_parts=6,
        wrap_name=False,
        wrap_path=True,
        wrap_target=True,
        subtoken_vocab_max_size=190000,
        target_vocab_max_size=27000,
    )


def get_preprocessing_config_astminer_code2vec_params(
    dataset_name: str, train: float, test: float, val: float
) -> PreprocessingConfig:
    return PreprocessingConfig(
        dataset_name=dataset_name,
        max_path_length=8,
        max_name_parts=5,
        max_target_parts=6,
        wrap_name=False,
        wrap_path=True,
        wrap_target=True,
        subtoken_vocab_max_size=190000,
        target_vocab_max_size=27000,
        train=train,
        test=test,
        val=val
    )


def get_code2seq_default_config(dataset_main_folder: str) -> Code2SeqConfig:
    encoder = EncoderConfig(embedding_size=128, rnn_size=128, use_bi_rnn=True, embedding_dropout=0.25, rnn_dropout=0.5)
    decoder = DecoderConfig(decoder_size=320, embedding_size=128, num_decoder_layers=1, rnn_dropout=0.5, beam_width=0)

    code2seq = Code2SeqConfig(
        train_data_path=join(dataset_main_folder, "train"),
        val_data_path=join(dataset_main_folder, "val"),
        test_data_path=join(dataset_main_folder, "test"),
        encoder=encoder,
        decoder=decoder,
        n_epochs=3000,
        patience=10,
        batch_size=512,
        test_batch_size=512,
        learning_rate=0.01,
        decay_gamma=0.95,
        clip_norm=5,
        max_context=200,
        random_context=True,
        shuffle_data=True,
    )
    return code2seq


def get_code2seq_test_config(dataset_main_folder: str) -> Code2SeqConfig:
    encoder = EncoderConfig(embedding_size=64, rnn_size=64, use_bi_rnn=True, embedding_dropout=0.25, rnn_dropout=0.5)
    decoder = DecoderConfig(decoder_size=120, embedding_size=64, num_decoder_layers=1, rnn_dropout=0.5, beam_width=0)

    code2seq = Code2SeqConfig(
        train_data_path=join(dataset_main_folder, "train"),
        val_data_path=join(dataset_main_folder, "val"),
        test_data_path=join(dataset_main_folder, "test"),
        encoder=encoder,
        decoder=decoder,
        n_epochs=5,
        patience=3,
        batch_size=10,
        test_batch_size=10,
        learning_rate=0.001,
        decay_gamma=0.95,
        clip_norm=5,
        max_context=200,
        random_context=True,
        shuffle_data=True,
    )
    return code2seq
