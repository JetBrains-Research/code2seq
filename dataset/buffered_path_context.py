import pickle
from dataclasses import dataclass
from typing import List, Dict, Tuple

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


def create_standard_bpc(
    config: PreprocessingConfig,
    vocab: Vocabulary,
    input_labels: List[List[int]],
    input_from_tokens: List[List[List[int]]],
    input_path_types: List[List[List[int]]],
    input_to_tokens: List[List[List[int]]],
) -> BufferedPathContext:
    if not (len(input_from_tokens) == len(input_path_types) == len(input_to_tokens)):
        raise ValueError(f"Unequal sizes of array with path parts")
    if len(input_labels) != len(input_from_tokens):
        raise ValueError(f"Number of labels is different to number of paths")

    contexts_per_label = [len(pc) for pc in input_from_tokens]
    total_num_contexts = sum(contexts_per_label)
    buffer_size = len(input_labels)

    def _reserve_array(max_seq_len: int, is_wrapped: bool, total_size) -> numpy.ndarray:
        return numpy.empty((max_seq_len + int(is_wrapped), total_size), dtype=numpy.int32)

    labels = _reserve_array(config.max_target_parts, config.wrap_target, buffer_size)
    from_tokens = _reserve_array(config.max_name_parts, config.wrap_name, total_num_contexts)
    path_types = _reserve_array(config.max_path_length, config.wrap_path, total_num_contexts)
    to_tokens = _reserve_array(config.max_name_parts, config.wrap_name, total_num_contexts)

    cur_path_idx = 0
    for sample in range(buffer_size):
        labels[:, sample] = _prepare_to_store(
            input_labels[sample], config.max_target_parts, vocab.label_to_id, config.wrap_target
        )
        for ft, pt, tt in zip(input_from_tokens[sample], input_path_types[sample], input_to_tokens[sample]):
            from_tokens[:, cur_path_idx] = _prepare_to_store(
                ft, config.max_name_parts, vocab.token_to_id, config.wrap_name
            )
            path_types[:, cur_path_idx] = _prepare_to_store(
                pt, config.max_path_length, vocab.type_to_id, config.wrap_path
            )
            to_tokens[:, cur_path_idx] = _prepare_to_store(
                tt, config.max_name_parts, vocab.token_to_id, config.wrap_name
            )
            cur_path_idx += 1

    contexts = {FROM_TOKEN: from_tokens, PATH_TYPES: path_types, TO_TOKEN: to_tokens}
    return BufferedPathContext(contexts, labels, contexts_per_label)


def _prepare_to_store(values: List[int], max_len: int, to_id: Dict, is_wrapped: bool) -> List[int]:
    used_len = min(len(values), max_len)
    result = [to_id[SOS]] + values[:used_len] if is_wrapped else values[:used_len]
    if used_len < max_len and is_wrapped:
        result.append(to_id[EOS])
        used_len += 1
    result += [to_id[PAD]] * (max_len - used_len)
    return result
