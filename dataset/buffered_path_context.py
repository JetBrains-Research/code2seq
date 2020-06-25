import pickle
from typing import List, Dict

from configs import PreprocessingConfig
from dataset import Vocabulary
from utils.common import PAD, SOS, EOS


class BufferedPathContext:
    """Class for storing buffered path contexts
    labels_array: list of shape [buffer_size; max_target_parts + 1]
    from\to_tokens_array: list of shape [buffer_size; paths_for_labels (unique per sample); max_name_parts + 1]
    path_types_array: list of shape [buffer_size; paths_for_labels (unique per sample); max_path_length + 1]

    +1 for SOS token, put EOS if enough space
    """

    def __init__(
        self, config: PreprocessingConfig, vocab: Vocabulary,
    ):
        self.vocab = vocab
        self.buffer_size = config.buffer_size
        self.max_target_parts = config.max_target_parts
        self.max_name_parts = config.max_name_parts
        self.max_path_length = config.max_path_length

        self.labels_array = []
        self.from_tokens_array = []
        self.path_types_array = []
        self.to_tokens_array = []

    @staticmethod
    def _prepare_to_store(values: List[int], max_len: int, to_id: Dict) -> List[int]:
        used_len = min(len(values), max_len)
        result = [to_id[SOS]] + values[:used_len]
        if used_len < max_len:
            result.append(to_id[EOS])
            used_len += 1
        result += [to_id[PAD]] * (max_len - used_len)
        return result

    def store_path_context(
        self, label: List[int], from_tokens: List[List[int]], path_types: List[List[int]], to_tokens: List[List[int]],
    ):
        if len(self.labels_array) == self.buffer_size:
            raise RuntimeError(f"Too many path contexts, create another buffered storage")
        if not (len(from_tokens) == len(path_types) == len(to_tokens)):
            raise ValueError(f"Unequal sizes of array with path parts")

        # store labels
        self.labels_array.append(self._prepare_to_store(label, self.max_target_parts, self.vocab.label_to_id))

        self.from_tokens_array.append([])
        self.path_types_array.append([])
        self.to_tokens_array.append([])
        for i in range(len(from_tokens)):
            # store from token
            self.from_tokens_array[-1].append(
                self._prepare_to_store(from_tokens[i], self.max_name_parts, self.vocab.token_to_id)
            )
            # store path types
            self.path_types_array[-1].append(
                self._prepare_to_store(path_types[i], self.max_path_length, self.vocab.type_to_id)
            )
            # store to token
            self.to_tokens_array[-1].append(
                self._prepare_to_store(to_tokens[i], self.max_name_parts, self.vocab.token_to_id)
            )

    def dump(self, path: str):
        with open(path, "wb") as pickle_file:
            pickle.dump(self, pickle_file)

    def __len__(self):
        return len(self.labels_array)
