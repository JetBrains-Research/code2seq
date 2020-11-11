from os.path import exists, join
from typing import List, Optional

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from configs.parts import PathContextConfig, ModelHyperParameters
from dataset import PathContextDataset, PathContextSample, PathContextBatch
from utils.common import TRAIN_HOLDOUT, TEST_HOLDOUT, VAL_HOLDOUT, DATA_FOLDER
from utils.vocabulary import Vocabulary


class PathContextDataModule(LightningDataModule):

    _train_dataset: PathContextDataset
    _val_dataset: PathContextDataset
    _test_dataset: PathContextDataset

    def __init__(
        self,
        dataset_name: str,
        vocabulary: Vocabulary,
        data_params: PathContextConfig,
        model_params: ModelHyperParameters,
        num_workers: int = 0,
    ):
        super().__init__()
        self._dataset_name = dataset_name
        self._data_config = data_params
        self._hyper_params = model_params
        self._num_workers = num_workers
        self._vocabulary = vocabulary

        self._dataset_dir = join(DATA_FOLDER, self._dataset_name)
        if not exists(self._dataset_dir):
            raise ValueError(f"There is no file in passed path ({self._dataset_dir})")
        self._train_data_file = join(self._dataset_dir, f"{self._dataset_name}.{TRAIN_HOLDOUT}.c2s")
        self._val_data_file = join(self._dataset_dir, f"{self._dataset_name}.{VAL_HOLDOUT}.c2s")
        self._test_data_file = join(self._dataset_dir, f"{self._dataset_name}.{TEST_HOLDOUT}.c2s")

    def prepare_data(self):
        if not exists(self._dataset_dir):
            raise ValueError(f"There is no file in passed path ({self._dataset_dir})")
        # TODO: download data from s3

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self._train_dataset = PathContextDataset(
                self._train_data_file,
                self._vocabulary,
                self._data_config,
                self._hyper_params.max_context,
                self._hyper_params.random_context,
            )
            self._val_dataset = PathContextDataset(
                self._val_data_file, self._vocabulary, self._data_config, self._hyper_params.max_context, False
            )
        else:
            self._test_dataset = PathContextDataset(
                self._test_data_file, self._vocabulary, self._data_config, self._hyper_params.max_context, False
            )

    @staticmethod
    def collate_wrapper(batch: List[PathContextSample]) -> PathContextBatch:
        return PathContextBatch(batch)

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(
            self._train_dataset,
            batch_size=self._hyper_params.batch_size,
            shuffle=self._hyper_params.shuffle_data,
            num_workers=self._num_workers,
            collate_fn=self.collate_wrapper,
            pin_memory=True,
        )

    def val_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(
            self._val_dataset,
            batch_size=self._hyper_params.test_batch_size,
            shuffle=False,
            num_workers=self._num_workers,
            collate_fn=self.collate_wrapper,
            pin_memory=True,
        )

    def test_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(
            self._test_dataset,
            batch_size=self._hyper_params.test_batch_size,
            shuffle=False,
            num_workers=self._num_workers,
            collate_fn=self.collate_wrapper,
            pin_memory=True,
        )

    def transfer_batch_to_device(self, batch: PathContextBatch, device: torch.device) -> PathContextBatch:
        batch.move_to_device(device)
        return batch
