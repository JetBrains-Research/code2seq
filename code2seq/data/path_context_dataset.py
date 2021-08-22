from os.path import exists
from random import shuffle
from typing import Dict, List, Optional

import torch
from commode_utils.filesystem import get_lines_offsets, get_line_by_offset
from omegaconf import DictConfig
from torch.utils.data import Dataset

from code2seq.data.path_context import LabeledPathContext, Path
from code2seq.data.vocabulary import Vocabulary


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
        if self._config.max_label_parts == 1:
            label = self.tokenize_class(raw_label, self._vocab.label_to_id)
        else:
            label = self.tokenize_label(raw_label, self._vocab.label_to_id, self._config.max_label_parts)

        # Tokenize paths
        try:
            paths = [self._get_path(raw_path.split(",")) for raw_path in raw_path_contexts]
        except ValueError as e:
            with open(self._log_file, "a") as f_out:
                f_out.write(f"Error parsing sample from line #{index}: {e}")
            return None

        return LabeledPathContext(label, paths)

    @staticmethod
    def tokenize_class(raw_class: str, vocab: Dict[str, int]) -> torch.Tensor:
        return torch.tensor([vocab[raw_class]], dtype=torch.long)

    @staticmethod
    def tokenize_label(raw_label: str, vocab: Dict[str, int], max_parts: Optional[int]) -> torch.Tensor:
        sublabels = raw_label.split(PathContextDataset._separator)
        max_parts = max_parts or len(sublabels)
        label_unk = vocab[Vocabulary.UNK]

        label = torch.full((max_parts + 1,), vocab[Vocabulary.PAD], dtype=torch.long)
        label[0] = vocab[Vocabulary.SOS]
        sub_tokens_ids = [vocab.get(st, label_unk) for st in sublabels[:max_parts]]
        label[1 : len(sub_tokens_ids) + 1] = torch.tensor(sub_tokens_ids)

        if len(sublabels) < max_parts:
            label[len(sublabels) + 1] = vocab[Vocabulary.EOS]

        return label

    @staticmethod
    def tokenize_token(token: str, vocab: Dict[str, int], max_parts: Optional[int]) -> torch.Tensor:
        sub_tokens = token.split(PathContextDataset._separator)
        max_parts = max_parts or len(sub_tokens)
        token_unk = vocab[Vocabulary.UNK]

        result = torch.full((max_parts,), vocab[Vocabulary.PAD], dtype=torch.long)
        sub_tokens_ids = [vocab.get(st, token_unk) for st in sub_tokens[:max_parts]]
        result[: len(sub_tokens_ids)] = torch.tensor(sub_tokens_ids)
        return result

    def _get_path(self, raw_path: List[str]) -> Path:
        return Path(
            from_token=self.tokenize_token(raw_path[0], self._vocab.token_to_id, self._config.max_token_parts),
            path_node=self.tokenize_token(raw_path[1], self._vocab.node_to_id, self._config.path_length),
            to_token=self.tokenize_token(raw_path[2], self._vocab.token_to_id, self._config.max_token_parts),
        )
