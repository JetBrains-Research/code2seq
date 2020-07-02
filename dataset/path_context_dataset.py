from math import ceil
from os import listdir
from os.path import exists, join
from typing import Dict, Tuple, List

import numpy
import torch
from torch.utils.data import IterableDataset

from dataset import BufferedPathContext
from utils.common import FROM_TOKEN, PATH_TYPES, TO_TOKEN


class PathContextDataset(IterableDataset):
    def __init__(self, path: str, max_context: int, random_context: bool, shuffle: bool, batch_size: int = 1):
        super().__init__()
        if not exists(path):
            raise ValueError(f"Path does not exist")
        self.max_context = max_context
        self.random_context = random_context
        self.shuffle = shuffle
        self.batch_size = batch_size

        buffered_files = listdir(path)
        buffered_files = sorted(buffered_files, key=lambda file: int(file.rsplit("_", 1)[1][:-4]))
        self._buffered_files_paths = [join(path, bf) for bf in buffered_files]

        self._total_n_samples = 0
        for filepath in self._buffered_files_paths:
            buf_path_context = BufferedPathContext.load(filepath)
            self._total_n_samples += len(buf_path_context)

        # each worker use data from _cur_file_idx and until it reaches _end_file_idx
        self._cur_file_idx = None
        self._end_file_idx = None
        self._cur_buffered_path_context = None

    def _prepare_buffer(self, file_idx: int) -> None:
        assert file_idx < len(self._buffered_files_paths)
        self._cur_buffered_path_context = BufferedPathContext.load(self._buffered_files_paths[file_idx])
        self._order = numpy.arange(len(self._cur_buffered_path_context))
        if self.shuffle:
            self._order = numpy.random.permutation(self._order)
        self._cur_sample_idx = 0

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            self._cur_file_idx = 0
            self._end_file_idx = len(self._buffered_files_paths)
        else:
            worker_id = worker_info.id
            per_worker = int(ceil(len(self._buffered_files_paths) / float(worker_info.num_workers)))
            self._cur_file_idx = per_worker * worker_id
            self._end_file_idx = min(self._cur_file_idx + per_worker, len(self._buffered_files_paths))
        return self

    def __next__(self) -> Tuple[Dict[str, numpy.ndarray], numpy.ndarray, int]:
        if self._cur_buffered_path_context is None:
            if self._cur_file_idx >= self._end_file_idx:
                raise StopIteration()
            else:
                self._prepare_buffer(self._cur_file_idx)
        if self._cur_sample_idx == len(self._order):
            self._cur_file_idx += 1
            if self._cur_file_idx >= self._end_file_idx:
                raise StopIteration()
            self._prepare_buffer(self._cur_file_idx)
        context, label, paths_for_label = self._cur_buffered_path_context[self._order[self._cur_sample_idx]]

        # select max_context paths from sample
        context_idx = numpy.arange(paths_for_label)
        if self.random_context:
            context_idx = numpy.random.permutation(context_idx)
        paths_for_label = min(self.max_context, paths_for_label)
        context_idx = context_idx[:paths_for_label]
        for key in [FROM_TOKEN, PATH_TYPES, TO_TOKEN]:
            context[key] = context[key][:, context_idx]

        self._cur_sample_idx += 1
        return context, label, paths_for_label

    def __len__(self):
        # Since dataloader for IterableDataset doesn't compute length with respect to batch size
        # we do it here manually
        # https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
        return ceil(self._total_n_samples / self.batch_size)


def collate_path_contexts(
    samples: List[Tuple[Dict[str, numpy.ndarray], numpy.ndarray, int]]
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, List[int]]:
    from_tokens = [torch.tensor(sample[0][FROM_TOKEN]) for sample in samples]
    path_types = [torch.tensor(sample[0][PATH_TYPES]) for sample in samples]
    to_tokens = [torch.tensor(sample[0][TO_TOKEN]) for sample in samples]
    paths_for_label = [sample[2] for sample in samples]
    labels = [torch.tensor(sample[1]) for sample in samples]
    return (
        {
            FROM_TOKEN: torch.cat(from_tokens, dim=-1),
            PATH_TYPES: torch.cat(path_types, dim=-1),
            TO_TOKEN: torch.cat(to_tokens, dim=-1),
        },
        torch.cat(labels, dim=-1),
        paths_for_label,
    )
