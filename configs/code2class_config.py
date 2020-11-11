from dataclasses import dataclass

from configs.parts import PathContextConfig, ModelHyperParameters, EncoderConfig, ClassifierConfig, ContextDescription


@dataclass(frozen=True)
class Code2ClassConfig:
    path_context_processing = PathContextConfig(
        token_description=ContextDescription(max_parts=5, is_wrapped=False, is_splitted=True, vocab_size=190000),
        path_description=ContextDescription(max_parts=9, is_wrapped=False, is_splitted=True,),
        target_description=ContextDescription(max_parts=1, is_wrapped=False, is_splitted=False, vocab_size=27000),
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
class Code2ClassTestConfig(Code2ClassConfig):
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
