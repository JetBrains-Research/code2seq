from configs import Code2SeqConfig
from model import Code2Seq
from model.modules import TypedPathEncoder, PathEncoder
from utils.common import PAD
from utils.vocabulary import Vocabulary


class TypedCode2Seq(Code2Seq):
    def __init__(self, config: Code2SeqConfig, vocabulary: Vocabulary):
        assert (
            vocabulary.type_to_id is not None
        ), "You need to store type to id dict in vocabulary for using typed code2seq"
        super().__init__(config, vocabulary)

    def _get_encoder(self) -> PathEncoder:
        return TypedPathEncoder(
            self._config.encoder_config,
            self._config.decoder_config.decoder_size,
            len(self._vocabulary.token_to_id),
            self._vocabulary.token_to_id[PAD],
            len(self._vocabulary.node_to_id),
            self._vocabulary.node_to_id[PAD],
            len(self._vocabulary.type_to_id),
            self._vocabulary.type_to_id[PAD],
        )
