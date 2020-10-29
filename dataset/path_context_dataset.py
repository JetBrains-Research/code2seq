from os.path import exists
from typing import List, Dict

import numpy
from torch.utils.data import Dataset

from configs.parts import DataProcessingConfig
from dataset.data_classes import PathContextSample
from utils.common import FROM_TOKEN, TO_TOKEN, PATH_TYPES
from utils.vocabulary import Vocabulary
from utils.converting import string_to_wrapped_numpy


class PathContextDataset(Dataset):

    _separator = "|"

    def __init__(
        self,
        data_path: str,
        vocabulary: Vocabulary,
        config: DataProcessingConfig,
        max_context: int,
        random_context: bool,
    ):
        assert exists(data_path), f"Can't find file with data: {data_path}"
        self._vocab = vocabulary
        self._config = config
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

        self._context_fields = [
            (
                FROM_TOKEN,
                self._vocab.token_to_id,
                self._config.split_names,
                self._config.max_name_parts,
                self._config.wrap_name,
            ),
            (PATH_TYPES, self._vocab.type_to_id, True, self._config.max_path_length, self._config.wrap_path),
            (
                TO_TOKEN,
                self._vocab.token_to_id,
                self._config.split_names,
                self._config.max_name_parts,
                self._config.wrap_name,
            ),
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
        from_token, path_types, to_token = context.split(",")
        return {
            FROM_TOKEN: from_token,
            PATH_TYPES: path_types,
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
        wrapped_label = string_to_wrapped_numpy(
            str_label,
            self._vocab.label_to_id,
            self._config.split_target,
            self._config.max_target_parts,
            self._config.wrap_target,
        )

        # convert each context to list of ints and then wrap into numpy array
        contexts = {}
        for key, _, _, max_length, is_wrapped in self._context_fields:
            size = max_length + (1 if is_wrapped else 0)
            contexts[key] = numpy.empty((size, n_contexts), dtype=numpy.int32)
        for i, context_idx in enumerate(context_indexes):
            splitted_context = self._split_context(str_contexts[context_idx])
            for key, to_id, is_split, max_length, is_wrapped in self._context_fields:
                contexts[key][:, [i]] = string_to_wrapped_numpy(
                    splitted_context[key], to_id, is_split, max_length, is_wrapped
                )

        return PathContextSample(contexts=contexts, label=wrapped_label, n_contexts=n_contexts)
