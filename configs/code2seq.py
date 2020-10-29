from dataclasses import dataclass

from configs.parts import DataProcessingConfig, ModelHyperParameters, EncoderConfig, DecoderConfig


@dataclass(frozen=True)
class Code2SeqConfig(dataclass):
    data_processing = DataProcessingConfig(
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
    decoder_config = DecoderConfig(
        decoder_size=320, embedding_size=128, num_decoder_layers=1, rnn_dropout=0.5, teacher_forcing=1, beam_width=0
    )


@dataclass(frozen=True)
class Code2SeqTestConfig(dataclass):
    data_processing = DataProcessingConfig(
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
    decoder_config = DecoderConfig(
        decoder_size=120, embedding_size=64, num_decoder_layers=1, rnn_dropout=0.5, teacher_forcing=1, beam_width=0
    )
