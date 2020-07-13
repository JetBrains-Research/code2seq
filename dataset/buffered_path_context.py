import pickle
from dataclasses import dataclass, asdict
from itertools import chain
from typing import List, Dict, Tuple, Iterable

import numpy

from utils.common import PAD, SOS, EOS


@dataclass(frozen=True)
class ConvertParameters:
    max_length: int
    is_wrapped: bool
    to_id: Dict[str, int]


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
        input_labels: Tuple[List[List[int]], ConvertParameters],
        input_contexts: Dict[str, Tuple[List[List[List[int]]], ConvertParameters]],
    ) -> "BufferedPathContext":
        ctx_sizes = [len(ctx[0]) for ctx in input_contexts.values()]
        if not all([ctx_sz == ctx_sizes[0] for ctx_sz in ctx_sizes]):
            raise ValueError(f"Unequal sizes of array with path parts")
        if len(input_labels[0]) != ctx_sizes[0]:
            raise ValueError(f"Number of labels is different to number of paths")

        rnd_context = input_contexts[list(input_contexts.keys())[0]]
        contexts_per_label = [len(ctx) for ctx in rnd_context[0]]
        n_contexts = sum(contexts_per_label)

        labels = BufferedPathContext._list_to_numpy_array(
            input_labels[0], len(input_labels[0]), **asdict(input_labels[1])
        )
        contexts = {}
        for ctx_name, ctx in input_contexts.items():
            contexts[ctx_name] = BufferedPathContext._list_to_numpy_array(
                chain.from_iterable(ctx[0]), n_contexts, **asdict(ctx[1])
            )

        return BufferedPathContext(contexts, labels, contexts_per_label)

    @staticmethod
    def _list_to_numpy_array(
        values: Iterable[List[int]], total_size: int, max_length: int, is_wrapped: bool, to_id: Dict
    ) -> numpy.ndarray:
        result = numpy.full((max_length + int(is_wrapped), total_size), to_id[PAD], dtype=numpy.int32)
        start_idx = 0
        if is_wrapped:
            result[0, :] = to_id[SOS]
            start_idx = 1
        for pos, sample in enumerate(values):
            used_len = min(len(sample), max_length)
            result[start_idx : used_len + start_idx, pos] = sample[:used_len]
            if used_len < max_length and is_wrapped:
                result[used_len + start_idx, pos] = to_id[EOS]
        return result
