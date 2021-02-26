from omegaconf import DictConfig

from code2seq.model import Code2Seq
from code2seq.model.modules import TypedPathEncoder, PathEncoder
from code2seq.utils.vocabulary import Vocabulary, PAD


class TypedCode2Seq(Code2Seq):
    def __init__(self, config: DictConfig, vocabulary: Vocabulary):
        super().__init__(config, vocabulary)

    def _get_encoder(self) -> PathEncoder:
        assert (
            self._vocabulary.type_to_id is not None
        ), "You need to store type to id dict in vocabulary for using typed code2seq"
        return TypedPathEncoder(
            self._config.encoder,
            self._config.decoder.decoder_size,
            len(self._vocabulary.token_to_id),
            self._vocabulary.token_to_id[PAD],
            len(self._vocabulary.node_to_id),
            self._vocabulary.node_to_id[PAD],
            len(self._vocabulary.type_to_id),
            self._vocabulary.type_to_id[PAD],
        )
