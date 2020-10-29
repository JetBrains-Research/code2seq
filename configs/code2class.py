from dataclasses import dataclass

from configs.parts import DataProcessingConfig, ModelHyperParameters, EncoderConfig, ClassifierConfig


@dataclass(frozen=True)
class Code2ClassConfig:
    data_processing = DataProcessingConfig(
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
    hyper_parameters = ModelHyperParameters(
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
    encoder_config = EncoderConfig(
        embedding_size=128, rnn_size=128, use_bi_rnn=True, embedding_dropout=0.25, rnn_num_layers=1, rnn_dropout=0.5
    )
    classifier_config = ClassifierConfig(
        n_hidden_layers=2, hidden_size=128, classifier_input_size=256, activation="relu"
    )


@dataclass(frozen=True)
class Code2ClassTestConfig:
    data_processing = DataProcessingConfig(
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
    hyper_parameters = ModelHyperParameters(
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
    encoder_config = EncoderConfig(
        embedding_size=64, rnn_size=64, use_bi_rnn=True, embedding_dropout=0.25, rnn_num_layers=1, rnn_dropout=0.5
    )
    classifier_config = ClassifierConfig(
        n_hidden_layers=2, activation="relu", hidden_size=64, classifier_input_size=120
    )
