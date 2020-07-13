import pickle
from dataclasses import dataclass
from itertools import chain
from typing import List, Dict, Tuple, Iterable

import numpy

from configs import PreprocessingConfig
from dataset import Vocabulary
from utils.common import PAD, SOS, EOS, FROM_TOKEN, PATH_TYPES, TO_TOKEN


@dataclass
class BufferedPathContext:
    """Class for storing buffered path contexts.

    contexts: dictionary for describing context each element is numpy array with shape
        [max size + 1; buffer_size * n_contexts (unique per sample)]
    labels: labels for given contexts[max_target_parts + 1; buffer size]
    contexts_per_label: list [buffer size] -- number of paths for each label

    +1 for SOS token, put EOS if enough space
    """

    contexts: Dict[str, numpy.ndarray]
    labels: numpy.ndarray
    contexts_per_label: List[int]

    def __post_init__(self):
        self._end_idx = numpy.cumsum(self.contexts_per_label).tolist()
        self._start_idx = [0] + self._end_idx[:-1]

    def __len__(self):
        return len(self.contexts_per_label)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, numpy.ndarray], numpy.ndarray, int]:
        path_slice = slice(self._start_idx[idx], self._end_idx[idx])
        item_contexts = {}
        for k, v in self.contexts.items():
            item_contexts[k] = v[:, path_slice]
        return item_contexts, self.labels[:, [idx]], self.contexts_per_label[idx]

    def dump(self, path: str):
        with open(path, "wb") as pickle_file:
            pickle.dump((self.contexts, self.labels, self.contexts_per_label), pickle_file)

    @staticmethod
    def load(path: str):
        with open(path, "rb") as pickle_file:
            data = pickle.load(pickle_file)
        if not isinstance(data, tuple) and len(data) != 3:
            raise RuntimeError("Incorrect data inside pickled file")
        return BufferedPathContext(*data)

    @staticmethod
    def create_from_lists(
        config: PreprocessingConfig,
        vocab: Vocabulary,
        input_labels: List[List[int]],
        input_from_tokens: List[List[List[int]]],
        input_path_types: List[List[List[int]]],
        input_to_tokens: List[List[List[int]]],
    ) -> "BufferedPathContext":
        if not (len(input_from_tokens) == len(input_path_types) == len(input_to_tokens)):
            raise ValueError(f"Unequal sizes of array with path parts")
        if len(input_labels) != len(input_from_tokens):
            raise ValueError(f"Number of labels is different to number of paths")

        contexts_per_label = [len(pc) for pc in input_from_tokens]
        n_contexts = sum(contexts_per_label)
        buffer_size = len(input_labels)

        labels = BufferedPathContext._list_to_numpy_array(
            input_labels, buffer_size, config.max_target_parts, config.wrap_target, vocab.label_to_id
        )
        from_tokens = BufferedPathContext._list_to_numpy_array(
            chain.from_iterable(input_from_tokens),
            n_contexts,
            config.max_name_parts,
            config.wrap_name,
            vocab.token_to_id,
        )
        path_types = BufferedPathContext._list_to_numpy_array(
            chain.from_iterable(input_path_types),
            n_contexts,
            config.max_path_length,
            config.wrap_path,
            vocab.type_to_id,
        )
        to_tokens = BufferedPathContext._list_to_numpy_array(
            chain.from_iterable(input_to_tokens),
            n_contexts,
            config.max_name_parts,
            config.wrap_name,
            vocab.token_to_id,
        )

        contexts = {FROM_TOKEN: from_tokens, PATH_TYPES: path_types, TO_TOKEN: to_tokens}
        return BufferedPathContext(contexts, labels, contexts_per_label)

    @staticmethod
    def _list_to_numpy_array(
        values: Iterable[List[int]], total_size: int, max_len: int, is_wrapped: bool, to_id: Dict
    ) -> numpy.ndarray:
        result = numpy.full((max_len + int(is_wrapped), total_size), to_id[PAD], dtype=numpy.int32)
        start_idx = 0
        if is_wrapped:
            result[0, :] = to_id[SOS]
            start_idx = 1
        for pos, sample in enumerate(values):
            used_len = min(len(sample), max_len)
            result[start_idx : used_len + start_idx, pos] = sample[:used_len]
            if used_len < max_len and is_wrapped:
                result[used_len + start_idx, pos] = to_id[EOS]
        return result
