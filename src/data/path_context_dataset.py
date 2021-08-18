from os.path import exists
from random import shuffle
from typing import Dict, List, Optional

import torch
from commode_utils.filesystem import get_lines_offsets, get_line_by_offset
from omegaconf import DictConfig
from torch.utils.data import Dataset

from src.data.path_context import LabeledPathContext, Path
from src.data.vocabulary import Vocabulary


class PathContextDataset(Dataset):
    _log_file = "bad_samples.log"
    _separator = "|"

    def __init__(self, data_file: str, config: DictConfig, vocabulary: Vocabulary, random_context: bool):
        if not exists(data_file):
            raise ValueError(f"Can't find file with data: {data_file}")
        self._data_file = data_file
        self._config = config
        self._vocab = vocabulary
        self._random_context = random_context

        self._label_unk = vocabulary.label_to_id[vocabulary.UNK]

        self._line_offsets = get_lines_offsets(data_file)
        self._n_samples = len(self._line_offsets)

        open(self._log_file, "w").close()

    def __len__(self):
        return self._n_samples

    def __getitem__(self, index) -> Optional[LabeledPathContext]:
        raw_sample = get_line_by_offset(self._data_file, self._line_offsets[index])
        try:
            raw_label, *raw_path_contexts = raw_sample.split()
        except ValueError as e:
            with open(self._log_file, "a") as f_out:
                f_out.write(f"Error reading sample from line #{index}: {e}")
            return None

        # Choose paths for current data sample
        n_contexts = min(len(raw_path_contexts), self._config.max_context)
        if self._random_context:
            shuffle(raw_path_contexts)
        raw_path_contexts = raw_path_contexts[:n_contexts]

        # Tokenize label
        label = self._tokenize_label(raw_label)

        # Tokenize paths
        try:
            paths = [self._get_path(raw_path.split(",")) for raw_path in raw_path_contexts]
        except ValueError as e:
            with open(self._log_file, "a") as f_out:
                f_out.write(f"Error parsing sample from line #{index}: {e}")
            return None

        return LabeledPathContext(label, paths)

    def _tokenize_label(self, raw_label: str) -> torch.Tensor:
        label = torch.full((self._config.max_label_parts + 1,), self._vocab.label_to_id[self._vocab.PAD])
        label[0] = self._vocab.label_to_id[self._vocab.SOS]
        sublabels = raw_label.split(self._separator)[: self._config.max_label_parts]
        label[1 : len(sublabels) + 1] = torch.tensor(
            [self._vocab.label_to_id.get(sl, self._label_unk) for sl in sublabels]
        )
        if len(sublabels) < self._config.max_label_parts:
            label[len(sublabels) + 1] = self._vocab.label_to_id[self._vocab.EOS]
        return label

    def _tokenize_token(self, token: str, vocab: Dict[str, int], max_parts: Optional[int]) -> torch.Tensor:
        sub_tokens = token.split(self._separator)
        max_parts = max_parts or len(sub_tokens)
        token_unk = vocab[self._vocab.UNK]

        result = torch.full((max_parts,), vocab[self._vocab.PAD], dtype=torch.long)
        sub_tokens_ids = [vocab.get(st, token_unk) for st in sub_tokens[:max_parts]]
        result[: len(sub_tokens_ids)] = torch.tensor(sub_tokens_ids)
        return result

    def _get_path(self, raw_path: List[str]) -> Path:
        return Path(
            from_token=self._tokenize_token(raw_path[0], self._vocab.token_to_id, self._config.max_token_parts),
            path_node=self._tokenize_token(raw_path[1], self._vocab.node_to_id, self._config.path_length),
            to_token=self._tokenize_token(raw_path[2], self._vocab.token_to_id, self._config.max_token_parts),
        )
