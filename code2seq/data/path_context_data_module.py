from os.path import exists, join, basename
from typing import List, Optional

import torch
from commode_utils.common import download_dataset
from commode_utils.vocabulary import build_from_scratch
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from code2seq.data.path_context import LabeledPathContext, BatchedLabeledPathContext
from code2seq.data.path_context_dataset import PathContextDataset
from code2seq.data.vocabulary import Vocabulary


class PathContextDataModule(LightningDataModule):
    _train = "train"
    _val = "val"
    _test = "test"

    _vocabulary: Optional[Vocabulary] = None

    def __init__(self, data_dir: str, config: DictConfig, is_class: bool = False):
        super().__init__()
        self._config = config
        self._data_dir = data_dir
        self._name = basename(data_dir)
        self._is_class = is_class

    @property
    def vocabulary(self) -> Vocabulary:
        if self._vocabulary is None:
            raise RuntimeError(f"Setup data module for initializing vocabulary")
        return self._vocabulary

    def prepare_data(self):
        if exists(self._data_dir):
            print(f"Dataset is already downloaded")
            return
        if "url" not in self._config:
            raise ValueError(f"Config doesn't contain url for, can't download it automatically")
        download_dataset(self._config.url, self._data_dir, self._name)

    def setup(self, stage: Optional[str] = None):
        if not exists(join(self._data_dir, Vocabulary.vocab_filename)):
            print("Can't find vocabulary, collect it from train holdout")
            build_from_scratch(join(self._data_dir, f"{self._train}.c2s"), Vocabulary)
        vocabulary_path = join(self._data_dir, Vocabulary.vocab_filename)
        self._vocabulary = Vocabulary(vocabulary_path, self._config.max_labels, self._config.max_tokens, self._is_class)

    @staticmethod
    def collate_wrapper(batch: List[Optional[LabeledPathContext]]) -> BatchedLabeledPathContext:
        return BatchedLabeledPathContext(batch)

    def _create_dataset(self, holdout_file: str, random_context: bool) -> PathContextDataset:
        if self._vocabulary is None:
            raise RuntimeError(f"Setup vocabulary before creating data loaders")
        return PathContextDataset(holdout_file, self._config, self._vocabulary, random_context)

    def _shared_dataloader(self, holdout: str) -> DataLoader:
        if self._vocabulary is None:
            raise RuntimeError(f"Setup vocabulary before creating data loaders")

        holdout_file = join(self._data_dir, f"{holdout}.c2s")
        random_context = self._config.random_context if holdout == self._train else False
        dataset = self._create_dataset(holdout_file, random_context)

        batch_size = self._config.batch_size if holdout == self._train else self._config.test_batch_size
        shuffle = holdout == self._train

        return DataLoader(
            dataset,
            batch_size,
            shuffle=shuffle,
            num_workers=self._config.num_workers,
            collate_fn=self.collate_wrapper,
            pin_memory=True,
        )

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return self._shared_dataloader(self._train)

    def val_dataloader(self, *args, **kwargs) -> DataLoader:
        return self._shared_dataloader(self._val)

    def test_dataloader(self, *args, **kwargs) -> DataLoader:
        return self._shared_dataloader(self._test)

    def transfer_batch_to_device(
        self, batch: BatchedLabeledPathContext, device: torch.device, dataloader_idx: int
    ) -> BatchedLabeledPathContext:
        batch.move_to_device(device)
        return batch
