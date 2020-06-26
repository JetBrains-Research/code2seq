import pickle
from os import listdir
from os.path import exists, join
from typing import Dict, Tuple, List

import numpy
import torch
from torch.utils.data import IterableDataset

from dataset import BufferedPathContext


class PathContextDataset(IterableDataset):
    def __init__(self, path: str, shuffle: bool):
        super().__init__()
        if not exists(path):
            raise ValueError(f"Path does not exist")
        self.shuffle = shuffle

        buffered_files = listdir(path)
        buffered_files = sorted(buffered_files, key=lambda file: int(file.rsplit("_", 1)[1][:-4]))
        self._buffered_files_paths = [join(path, bf) for bf in buffered_files]

        self._cur_file_idx = 0
        self._prepare_buffer(self._cur_file_idx)

    def _prepare_buffer(self, file_idx: int) -> None:
        assert file_idx < len(self._buffered_files_paths)
        with open(self._buffered_files_paths[file_idx], "rb") as pickle_file:
            self._cur_buffered_path_context: BufferedPathContext = pickle.load(pickle_file)
        self._order = range(len(self._cur_buffered_path_context))
        if self.shuffle:
            self._order = numpy.random.permutation(self._order)
        self._cur_sample_idx = 0

    def __iter__(self):
        return self

    def __next__(self) -> Tuple[Dict[str, numpy.ndarray], numpy.ndarray]:
        if self._cur_sample_idx == len(self._order):
            self._cur_file_idx += 1
            if self._cur_file_idx == len(self._buffered_files_paths):
                raise StopIteration()
            self._prepare_buffer(self._cur_file_idx)
        sample = self._cur_buffered_path_context[self._order[self._cur_sample_idx]]
        self._cur_sample_idx += 1
        return sample


def collate_path_contexts(
    samples: List[Tuple[Dict[str, numpy.ndarray], numpy.ndarray]]
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    from_tokens = [torch.tensor(sample[0]["from_token"]) for sample in samples]
    path_types = [torch.tensor(sample[0]["path_types"]) for sample in samples]
    to_tokens = [torch.tensor(sample[0]["to_token"]) for sample in samples]
    paths_for_label = [sample[0]["paths_for_label"] for sample in samples]
    labels = [torch.tensor(sample[1]) for sample in samples]
    return (
        {
            "from_tokens": torch.cat(from_tokens, dim=-1),
            "path_types": torch.cat(path_types, dim=-1),
            "to_tokens": torch.cat(to_tokens, dim=-1),
            "paths_for_label": torch.tensor(paths_for_label),
        },
        torch.cat(labels, dim=-1),
    )
