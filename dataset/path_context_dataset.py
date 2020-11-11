from os.path import exists
from typing import Dict, List

import numpy
from torch.utils.data import Dataset

from configs.parts import PathContextConfig
from dataset.data_classes import PathContextSample, ContextField
from utils.common import FROM_TOKEN, TO_TOKEN, PATH_NODES
from utils.converting import strings_to_wrapped_numpy
from utils.vocabulary import Vocabulary


class PathContextDataset(Dataset):

    _separator = "|"

    def __init__(
        self, data_path: str, vocabulary: Vocabulary, config: PathContextConfig, max_context: int, random_context: bool,
    ):
        assert exists(data_path), f"Can't find file with data: {data_path}"
        self._max_context = max_context
        self._random_context = random_context
        self._data_path = data_path
        self._line_offsets = []
        cumulative_offset = 0
        with open(self._data_path, "r") as data_file:
            for line in data_file:
                self._line_offsets.append(cumulative_offset)
                cumulative_offset += len(line.encode(data_file.encoding))
        self._n_samples = len(self._line_offsets)

        self._target_vocabulary = vocabulary.label_to_id
        self._target_description = config.target_description

        self._context_fields: List[ContextField] = [
            ContextField(FROM_TOKEN, vocabulary.token_to_id, config.token_description),
            ContextField(PATH_NODES, vocabulary.node_to_id, config.path_description),
            ContextField(TO_TOKEN, vocabulary.token_to_id, config.token_description),
        ]

    def __len__(self):
        return self._n_samples

    def _read_line(self, index: int) -> str:
        with open(self._data_path, "r") as data_file:
            data_file.seek(self._line_offsets[index])
            line = data_file.readline().strip()
        return line

    @staticmethod
    def _split_context(context: str) -> Dict[str, str]:
        from_token, path_nodes, to_token = context.split(",")
        return {
            FROM_TOKEN: from_token,
            PATH_NODES: path_nodes,
            TO_TOKEN: to_token,
        }

    def __getitem__(self, index) -> PathContextSample:
        raw_sample = self._read_line(index)
        str_label, *str_contexts = raw_sample.split()

        # choose random paths
        n_contexts = min(len(str_contexts), self._max_context)
        context_indexes = numpy.arange(n_contexts)
        if self._random_context:
            numpy.random.shuffle(context_indexes)

        # convert string label to wrapped numpy array
        wrapped_label = strings_to_wrapped_numpy(
            [str_label],
            self._target_vocabulary,
            self._target_description.is_splitted,
            self._target_description.max_parts,
            self._target_description.is_wrapped,
        )

        # convert each context to list of ints and then wrap into numpy array
        splitted_contexts = [self._split_context(str_contexts[i]) for i in context_indexes]
        contexts = {}
        for context_field in self._context_fields:
            key, to_id, desc = context_field.name, context_field.to_id, context_field.description
            str_values = [_sc[key] for _sc in splitted_contexts]
            contexts[key] = strings_to_wrapped_numpy(
                str_values, to_id, desc.is_splitted, desc.max_parts, desc.is_wrapped
            )

        return PathContextSample(contexts=contexts, label=wrapped_label, n_contexts=n_contexts)
