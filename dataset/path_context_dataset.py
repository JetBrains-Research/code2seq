from math import ceil
from os.path import exists, join
from typing import Dict, Tuple, List

import numpy
import torch
from torch.utils.data import IterableDataset, DataLoader

from dataset import BufferedPathContext
from utils.common import FROM_TOKEN, PATH_TYPES, TO_TOKEN


class PathContextDataset(IterableDataset):

    _description_filename = "description.csv"

    def __init__(self, path: str, max_context: int, random_context: bool, shuffle: bool):
        super().__init__()
        if not exists(path):
            raise ValueError(f"Path does not exist")
        if not exists(join(path, self._description_filename)):
            raise ValueError(f"Can't find description file with name {self._description_filename}")

        self._buffered_files_paths = []
        self._total_n_samples = 0
        with open(join(path, self._description_filename)) as desc_file:
            header = desc_file.readline()  # read header
            assert header.strip() == "id,filename,n_samples,n_paths"
            for line in desc_file:
                file_id, filename, n_samples, n_paths = line.strip().split(",")
                self._buffered_files_paths.insert(int(file_id), join(path, filename))
                self._total_n_samples += int(n_samples)

        self.max_context = max_context
        self.random_context = random_context
        self.shuffle = shuffle

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

    def get_n_samples(self):
        return self._total_n_samples


class PathContextBatch:
    def __init__(self, samples: List[Tuple[Dict[str, numpy.ndarray], numpy.ndarray, int]]):
        self.context = {
            FROM_TOKEN: torch.cat(
                [torch.tensor(sample[0][FROM_TOKEN], dtype=torch.long) for sample in samples], dim=-1
            ),
            PATH_TYPES: torch.cat(
                [torch.tensor(sample[0][PATH_TYPES], dtype=torch.long) for sample in samples], dim=-1
            ),
            TO_TOKEN: torch.cat([torch.tensor(sample[0][TO_TOKEN], dtype=torch.long) for sample in samples], dim=-1),
        }

        self.labels = torch.cat([torch.tensor(sample[1], dtype=torch.long) for sample in samples], dim=-1)
        self.contexts_per_label = [sample[2] for sample in samples]

    def pin_memory(self):
        for k in self.context:
            self.context[k] = self.context[k].pin_memory()
        self.labels = self.labels.pin_memory()
        return self

    @staticmethod
    def collate_wrapper(batch: List[Tuple[Dict[str, numpy.ndarray], numpy.ndarray, int]]) -> "PathContextBatch":
        return PathContextBatch(batch)


def create_dataloader(
    path: str, max_context: int, random_context: bool, shuffle: bool, batch_size: int, n_workers: int,
) -> Tuple[DataLoader, int]:
    dataset = PathContextDataset(path, max_context, random_context, shuffle)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=PathContextBatch.collate_wrapper,
        num_workers=n_workers,
        pin_memory=True,
    )
    return dataloader, dataset.get_n_samples()
