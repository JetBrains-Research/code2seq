from omegaconf import DictConfig

from src.data.vocabulary import TypedVocabulary
from src.model import Code2Seq
from src.model.modules import TypedPathEncoder, PathEncoder


class TypedCode2Seq(Code2Seq):
    def __init__(
        self,
        model_config: DictConfig,
        optimizer_config: DictConfig,
        vocabulary: TypedVocabulary,
        teacher_forcing: float = 0.0,
    ):
        super().__init__(model_config, optimizer_config, vocabulary, teacher_forcing)
        self._vocabulary = vocabulary

    def _get_encoder(self, config: DictConfig) -> PathEncoder:
        return TypedPathEncoder(
            config,
            len(self._vocabulary.token_to_id),
            self._vocabulary.token_to_id[TypedVocabulary.PAD],
            len(self._vocabulary.node_to_id),
            self._vocabulary.node_to_id[TypedVocabulary.PAD],
            len(self._vocabulary.type_to_id),
            self._vocabulary.type_to_id[TypedVocabulary.PAD],
        )
