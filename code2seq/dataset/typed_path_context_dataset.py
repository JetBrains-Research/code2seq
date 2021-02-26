from typing import Dict

from omegaconf import DictConfig

from code2seq.dataset import PathContextDataset
from code2seq.dataset.data_classes import ContextPart, FROM_TYPE, TO_TYPE, FROM_TOKEN, PATH_NODES, TO_TOKEN
from code2seq.utils.vocabulary import Vocabulary


class TypedPathContextDataset(PathContextDataset):
    def __init__(self, data_file_path: str, config: DictConfig, vocabulary: Vocabulary, random_context: bool):
        super().__init__(data_file_path, config, vocabulary, random_context)
        assert (
            vocabulary.type_to_id is not None
        ), "You need to store type to id dict in vocabulary for using typed path context dataset"

        self._context_parts += [
            ContextPart(FROM_TYPE, vocabulary.type_to_id, config.dataset.type),
            ContextPart(TO_TYPE, vocabulary.type_to_id, config.dataset.type),
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
