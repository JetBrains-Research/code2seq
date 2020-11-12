from omegaconf import DictConfig

from model import Code2Seq
from model.modules import TypedPathEncoder, PathEncoder
from utils.vocabulary import Vocabulary, PAD


class TypedCode2Seq(Code2Seq):
    def __init__(self, config: DictConfig, vocabulary: Vocabulary):
        super().__init__(config, vocabulary)

    def _get_encoder(self) -> PathEncoder:
        assert (
            self._vocabulary.type_to_id is not None
        ), "You need to store type to id dict in vocabulary for using typed code2seq"
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
