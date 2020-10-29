from os.path import exists
from typing import List, Dict

import numpy
from torch.utils.data import Dataset

from configs import PreprocessingConfig
from dataset.data_classes import PathContextSample
from utils.common import Vocabulary, FROM_TOKEN, TO_TOKEN, PATH_TYPES
from utils.converting import list_to_wrapped_numpy, str_to_list


class PathContextDataset(Dataset):

    _separator = "|"

    def __init__(self, data_path: str, vocabulary: Vocabulary, config: PreprocessingConfig, max_context: int):
        assert exists(data_path), f"Can't find file with data: {data_path}"
        self._vocab = vocabulary
        self._config = config
        self._max_context = max_context
        self._data_path = data_path
        self._line_offsets = []
        cumulative_offset = 0
        with open(self._data_path, "r") as data_file:
            for line in data_file:
                self._line_offsets.append(cumulative_offset)
                cumulative_offset += len(line.encode(data_file.encoding))
        self._n_samples = len(self._line_offsets)

        self._context_fields = [
            (FROM_TOKEN, self._vocab.token_to_id, self._config.max_name_parts, self._config.wrap_name),
            (PATH_TYPES, self._vocab.type_to_id, self._config.max_path_length, self._config.wrap_path),
            (TO_TOKEN, self._vocab.token_to_id, self._config.max_name_parts, self._config.wrap_name),
        ]

    def __len__(self):
        return self._n_samples

    def _read_line(self, index: int) -> str:
        with open(self._data_path, "r") as data_file:
            data_file.seek(self._line_offsets[index])
            line = data_file.readline().strip()
        return line

    def _context_to_list(self, context: str) -> Dict[str, List[int]]:
        from_token, path_types, to_token = context.split(",")
        return {
            FROM_TOKEN: str_to_list(from_token, self._vocab.token_to_id, self._config.split_names, self._separator),
            PATH_TYPES: str_to_list(path_types, self._vocab.type_to_id, True, self._separator),
            TO_TOKEN: str_to_list(to_token, self._vocab.token_to_id, self._config.split_names, self._separator),
        }

    def __getitem__(self, index) -> PathContextSample:
        raw_sample = self._read_line(index)
        str_label, *str_contexts = raw_sample.split()

        # choose random paths
        n_contexts = min(len(str_contexts), self._max_context)
        context_indexes = numpy.random.choice(len(str_contexts), n_contexts, replace=False)

        # convert string label to wrapped numpy array
        list_label = str_to_list(str_label, self._vocab.label_to_id, self._config.split_target, self._separator)
        wrapped_label = list_to_wrapped_numpy(
            list_label, self._vocab.label_to_id, self._config.max_target_parts, self._config.wrap_target
        )

        # convert each context to list of ints and then wrap into numpy array
        contexts = {}
        for key, _, max_length, is_wrapped in self._context_fields:
            size = max_length + (1 if is_wrapped else 0)
            contexts[key] = numpy.empty((size, n_contexts), dtype=numpy.int32)
        for i, context_idx in enumerate(context_indexes):
            list_context = self._context_to_list(str_contexts[context_idx])
            for key, to_id, max_length, is_wrapped in self._context_fields:
                contexts[key][:, [i]] = list_to_wrapped_numpy(list_context[key], to_id, max_length, is_wrapped)

        return PathContextSample(contexts=contexts, label=wrapped_label, n_contexts=n_contexts)
