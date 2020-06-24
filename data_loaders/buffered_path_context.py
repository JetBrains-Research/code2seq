from typing import List

import numpy

from data_loaders import Vocabulary
from utils.common import PAD, SOS, EOS


class BufferedPathContext:
    """Class for storing buffered path contexts
    labels_array: [max_target_parts + 1; buffer_size]
    from\to_tokens_array: [max_name_parts + 1; buffer_size; paths_per_label]
    path_types_array: [max_path_length + 1; buffer_size; paths_per_label]

    +1 for SOS token, put EOS if enough space
    """

    def __init__(
        self,
        buffer_size: int,
        max_target_parts: int,
        max_name_parts: int,
        max_path_length: int,
        paths_per_label: int,
        vocab: Vocabulary,
    ):
        self.vocab = vocab
        self.max_target_parts = max_target_parts
        self.max_name_parts = max_name_parts
        self.max_path_length = max_path_length
        self.paths_per_label = paths_per_label

        self.labels_array = numpy.full((max_target_parts + 1, buffer_size), self.vocab.label_to_id[PAD])
        self.from_tokens_array = numpy.full(
            (max_name_parts + 1, buffer_size, paths_per_label), self.vocab.token_to_id[PAD]
        )
        self.path_types_array = numpy.full(
            (max_path_length + 1, buffer_size, paths_per_label), self.vocab.type_to_id[PAD]
        )
        self.to_tokens_array = numpy.full(
            (max_name_parts + 1, buffer_size, paths_per_label), self.vocab.token_to_id[PAD]
        )

        self.labels_array[0, :] = self.vocab.label_to_id[SOS]
        self.from_tokens_array[0, :] = self.vocab.token_to_id[SOS]
        self.path_types_array[0, :] = self.vocab.type_to_id[SOS]
        self.to_tokens_array[0, :] = self.vocab.token_to_id[SOS]

    @staticmethod
    def _prepare_to_store(values: List[int], max_len: int, eos_token: int) -> List[int]:
        used_len = min(len(values), max_len)
        if used_len < max_len:
            values.append(eos_token)
            used_len += 1
        return values[:used_len]

    def store_path_context(
        self,
        pos: int,
        label: List[int],
        from_tokens: List[List[int]],
        path_types: List[List[int]],
        to_tokens: List[List[int]],
    ):
        if not (len(from_tokens) == len(path_types) == len(to_tokens) == self.paths_per_label):
            raise ValueError(f"Wrong number of path contexts, should be {self.paths_per_label}")

        # store labels
        store_labels = self._prepare_to_store(label, self.max_target_parts, self.vocab.label_to_id[EOS])
        self.labels_array[1 : len(store_labels) + 1, pos] = store_labels

        for i in range(self.paths_per_label):
            # store from token
            store_from_tokens = self._prepare_to_store(from_tokens[i], self.max_name_parts, self.vocab.token_to_id[EOS])
            self.from_tokens_array[1 : len(store_from_tokens) + 1, pos, i] = store_from_tokens
            # store path types
            store_path_types = self._prepare_to_store(path_types[i], self.max_path_length, self.vocab.type_to_id[EOS])
            self.path_types_array[1 : len(store_path_types) + 1, pos, i] = store_path_types
            # store to token
            store_to_token = self._prepare_to_store(to_tokens[i], self.max_name_parts, self.vocab.token_to_id[EOS])
            self.to_tokens_array[1 : len(store_to_token) + 1, pos, i] = store_to_token
