from os.path import join

from .code2class_config import Code2ClassConfig
from .code2seq_config import Code2SeqConfig
from .model_hyperparameters_config import ModelHyperparameters
from .modules_config import DecoderConfig, EncoderConfig, ClassifierConfig
from .preprocessing_config import PreprocessingConfig


def get_preprocessing_config_code2seq_params(dataset_name: str) -> PreprocessingConfig:
    return PreprocessingConfig(
        dataset_name=dataset_name,
        max_path_length=9,
        max_name_parts=5,
        max_target_parts=7,
        wrap_name=False,
        wrap_path=False,
        wrap_target=False,
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


def _get_default_hyperparams(dataset_main_folder: str) -> ModelHyperparameters:
    return ModelHyperparameters(
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


def get_code2seq_default_config(dataset_main_folder: str) -> Code2SeqConfig:
    encoder_config = EncoderConfig(
        embedding_size=128, rnn_size=128, use_bi_rnn=True, embedding_dropout=0.25, rnn_num_layers=1, rnn_dropout=0.5
    )
    decoder_config = DecoderConfig(
        decoder_size=320, embedding_size=128, num_decoder_layers=1, rnn_dropout=0.5, teacher_forcing=1, beam_width=0
    )

    hyperparams = _get_default_hyperparams(dataset_main_folder)
    return Code2SeqConfig(encoder_config=encoder_config, decoder_config=decoder_config, hyperparams=hyperparams)


def get_code2class_default_config(dataset_main_folder: str,) -> Code2ClassConfig:
    encoder_config = EncoderConfig(
        embedding_size=128, rnn_size=128, use_bi_rnn=True, embedding_dropout=0.25, rnn_num_layers=1, rnn_dropout=0.5
    )
    classifier_config = ClassifierConfig(
        n_hidden_layers=2, hidden_size=128, classifier_input_size=256, activation="relu"
    )
    hyperparams = _get_default_hyperparams(dataset_main_folder)
    return Code2ClassConfig(encoder_config=encoder_config, classifier_config=classifier_config, hyperparams=hyperparams)


def _get_test_hyperparams(dataset_main_folder: str) -> ModelHyperparameters:
    return ModelHyperparameters(
        train_data_path=join(dataset_main_folder, "train"),
        val_data_path=join(dataset_main_folder, "val"),
        test_data_path=join(dataset_main_folder, "test"),
        n_epochs=5,
        patience=3,
        batch_size=10,
        test_batch_size=10,
        learning_rate=0.01,
        weight_decay=0,
        decay_gamma=0.95,
        clip_norm=5,
        max_context=200,
        random_context=True,
        shuffle_data=True,
    )


def get_code2seq_test_config(dataset_main_folder: str) -> Code2SeqConfig:
    encoder_config = EncoderConfig(
        embedding_size=64, rnn_size=64, use_bi_rnn=True, embedding_dropout=0.25, rnn_num_layers=1, rnn_dropout=0.5
    )
    decoder_config = DecoderConfig(
        decoder_size=120, embedding_size=64, num_decoder_layers=1, rnn_dropout=0.5, teacher_forcing=1, beam_width=0
    )
    hyperparams = _get_test_hyperparams(dataset_main_folder)
    return Code2SeqConfig(encoder_config=encoder_config, decoder_config=decoder_config, hyperparams=hyperparams)


def get_code2class_test_config(dataset_main_folder: str,) -> Code2ClassConfig:
    encoder_config = EncoderConfig(
        embedding_size=64, rnn_size=64, use_bi_rnn=True, embedding_dropout=0.25, rnn_num_layers=1, rnn_dropout=0.5
    )
    classifier_config = ClassifierConfig(
        n_hidden_layers=2, activation="relu", hidden_size=64, classifier_input_size=120
    )
    hyperparams = _get_test_hyperparams(dataset_main_folder)
    return Code2ClassConfig(encoder_config=encoder_config, classifier_config=classifier_config, hyperparams=hyperparams)
