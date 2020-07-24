from os.path import join
from typing import Tuple

from .preprocessing_config import PreprocessingConfig
from .code2seq_config import Code2SeqConfig, DecoderConfig, EncoderConfig
from .code2class_config import ClassifierConfig
from .base_code_config import BaseCodeModelConfig


def get_preprocessing_config_code2seq_params(dataset_name: str) -> PreprocessingConfig:
    return PreprocessingConfig(
        dataset_name=dataset_name,
        max_path_length=9,
        max_name_parts=5,
        max_target_parts=6,
        wrap_name=False,
        wrap_path=False,
        wrap_target=True,
        split_target=True,
        split_names=True,
        subtoken_vocab_max_size=190000,
        target_vocab_max_size=27000,
    )


def get_preprocessing_config_code2class_params(dataset_name: str) -> PreprocessingConfig:
    return PreprocessingConfig(
        dataset_name=dataset_name,
        max_path_length=9,
        max_name_parts=5,
        max_target_parts=1,
        wrap_name=False,
        wrap_path=False,
        wrap_target=False,
        split_target=False,
        split_names=True,
        subtoken_vocab_max_size=190000,
        target_vocab_max_size=27000,
    )


def _get_default_base(dataset_main_folder: str) -> BaseCodeModelConfig:
    return BaseCodeModelConfig(
        train_data_path=join(dataset_main_folder, "train"),
        val_data_path=join(dataset_main_folder, "val"),
        test_data_path=join(dataset_main_folder, "test"),
        n_epochs=3000,
        patience=10,
        batch_size=512,
        test_batch_size=512,
        learning_rate=0.01,
        weight_decay=0,
        decay_gamma=0.95,
        clip_norm=5,
        max_context=200,
        random_context=True,
        shuffle_data=True,
    )


def get_code2seq_default_config(dataset_main_folder: str) -> Tuple[BaseCodeModelConfig, EncoderConfig, DecoderConfig]:
    encoder = EncoderConfig(
        embedding_size=128, rnn_size=128, use_bi_rnn=True, embedding_dropout=0.25, rnn_num_layers=1, rnn_dropout=0.5
    )
    decoder = DecoderConfig(
        decoder_size=320, embedding_size=128, num_decoder_layers=1, rnn_dropout=0.5, teacher_forcing=1, beam_width=0
    )

    base = _get_default_base(dataset_main_folder)
    return base, encoder, decoder


def get_code2class_default_config(
    dataset_main_folder: str,
) -> Tuple[BaseCodeModelConfig, EncoderConfig, ClassifierConfig]:
    encoder = EncoderConfig(
        embedding_size=128, rnn_size=128, use_bi_rnn=True, embedding_dropout=0.25, rnn_num_layers=1, rnn_dropout=0.5
    )
    decoder = ClassifierConfig(n_hidden_layers=5, hidden_size=128, classifier_input_size=256, activation="relu")
    base = _get_default_base(dataset_main_folder)
    return base, encoder, decoder


def _get_test_base(dataset_main_folder: str) -> BaseCodeModelConfig:
    return BaseCodeModelConfig(
        train_data_path=join(dataset_main_folder, "train"),
        val_data_path=join(dataset_main_folder, "val"),
        test_data_path=join(dataset_main_folder, "test"),
        n_epochs=5,
        patience=3,
        batch_size=25,
        test_batch_size=10,
        learning_rate=0.01,
        weight_decay=0,
        decay_gamma=0.95,
        clip_norm=5,
        max_context=100,
        random_context=True,
        shuffle_data=True,
    )


def get_code2seq_test_config(dataset_main_folder: str) -> Tuple[BaseCodeModelConfig, EncoderConfig, DecoderConfig]:
    encoder = EncoderConfig(
        embedding_size=64, rnn_size=64, use_bi_rnn=True, embedding_dropout=0.25, rnn_num_layers=1, rnn_dropout=0.5
    )
    decoder = DecoderConfig(
        decoder_size=120, embedding_size=64, num_decoder_layers=1, rnn_dropout=0.5, teacher_forcing=1, beam_width=0
    )
    base = _get_test_base(dataset_main_folder)
    return base, encoder, decoder


def get_code2class_test_config(dataset_main_folder: str) -> Tuple[BaseCodeModelConfig, EncoderConfig, ClassifierConfig]:
    encoder = EncoderConfig(
        embedding_size=64, rnn_size=64, use_bi_rnn=True, embedding_dropout=0.25, rnn_num_layers=1, rnn_dropout=0.5
    )
    decoder = ClassifierConfig(n_hidden_layers=2, activation="relu", hidden_size=12, classifier_input_size=24)
    base = _get_test_base(dataset_main_folder)
    return base, encoder, decoder
