from typing import Dict

from configs.parts import TypedPathContextConfig
from dataset import PathContextDataset
from dataset.data_classes import ContextField
from utils.common import FROM_TYPE, TO_TYPE, FROM_TOKEN, PATH_NODES, TO_TOKEN
from utils.vocabulary import Vocabulary


class TypedPathContextDataset(PathContextDataset):
    def __init__(
        self,
        data_path: str,
        vocabulary: Vocabulary,
        config: TypedPathContextConfig,
        max_context: int,
        random_context: bool,
    ):
        super().__init__(data_path, vocabulary, config, max_context, random_context)
        assert (
            vocabulary.type_to_id is not None
        ), "You need to store type to id dict in vocabulary for using typed path context dataset"

        self._context_fields += [
            ContextField(FROM_TYPE, vocabulary.type_to_id, config.type_description),
            ContextField(TO_TYPE, vocabulary.type_to_id, config.type_description),
        ]

    @staticmethod
    def _split_context(context: str) -> Dict[str, str]:
        from_type, from_token, path_nodes, to_token, to_type = context.split(",")
        return {
            FROM_TYPE: from_type,
            FROM_TOKEN: from_token,
            PATH_NODES: path_nodes,
            TO_TOKEN: to_token,
            TO_TYPE: to_type,
        }
