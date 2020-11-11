from dataclasses import dataclass

from configs import Code2SeqConfig, Code2SeqTestConfig
from configs.parts import TypedPathContextConfig, ContextDescription, ModelHyperParameters, EncoderConfig, DecoderConfig


@dataclass(frozen=True)
class TypedCode2SeqConfig(Code2SeqConfig):
    data_processing = TypedPathContextConfig(
        token_description=ContextDescription(max_parts=5, is_wrapped=False, is_splitted=True, vocab_size=190000),
        path_description=ContextDescription(max_parts=9, is_wrapped=False, is_splitted=True,),
        target_description=ContextDescription(max_parts=7, is_wrapped=False, is_splitted=True, vocab_size=27000),
        type_description=ContextDescription(max_parts=5, is_wrapped=False, is_splitted=True,),
    )


@dataclass(frozen=True)
class TypedCode2SeqTestConfig(TypedCode2SeqConfig):
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
