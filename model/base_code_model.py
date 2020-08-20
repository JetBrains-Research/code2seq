from abc import abstractmethod
from math import ceil
from typing import Tuple, Dict, List

import torch
from pytorch_lightning.core.lightning import LightningModule
from torch.optim import Adam, Optimizer, SGD
from torch.optim.lr_scheduler import _LRScheduler, LambdaLR
from torch.utils.data import DataLoader

from configs import ModelHyperparameters
from dataset import Vocabulary, create_dataloader, PathContextBatch


class BaseCodeModel(LightningModule):
    def __init__(
        self, hyperparams: ModelHyperparameters, vocab: Vocabulary, num_workers: int = 0,
    ):
        super().__init__()
        self.hyperparams = hyperparams
        self.vocab = vocab
        self.num_workers = num_workers

    @abstractmethod
    def _general_epoch_end(self, outputs: List[Dict], loss_key: str, group: str) -> Dict:
        pass

    # ===== OPTIMIZERS =====

    def configure_optimizers(self) -> Tuple[List[Optimizer], List[_LRScheduler]]:
        if self.hyperparams.optimizer == "Momentum":
            # using the same momentum value as in original realization by Alon
            optimizer = SGD(
                self.parameters(),
                self.hyperparams.learning_rate,
                momentum=0.95,
                nesterov=self.hyperparams.nesterov,
                weight_decay=self.hyperparams.weight_decay,
            )
        elif self.hyperparams.optimizer == "Adam":
            optimizer = Adam(
                self.parameters(), self.hyperparams.learning_rate, weight_decay=self.hyperparams.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer name: {self.hyperparams.optimizer}, try one of: Adam, Momentum")
        scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: self.hyperparams.decay_gamma ** epoch)
        return [optimizer], [scheduler]

    # ===== DATALOADERS BLOCK =====

    def train_dataloader(self) -> DataLoader:
        dataloader, n_samples = create_dataloader(
            self.hyperparams.train_data_path,
            self.hyperparams.max_context,
            self.hyperparams.random_context,
            self.hyperparams.shuffle_data,
            self.hyperparams.batch_size,
            self.num_workers,
        )
        print(f"\napproximate number of steps for train is {ceil(n_samples / self.hyperparams.batch_size)}")
        return dataloader

    def val_dataloader(self) -> DataLoader:
        dataloader, n_samples = create_dataloader(
            self.hyperparams.val_data_path,
            self.hyperparams.max_context,
            False,
            False,
            self.hyperparams.test_batch_size,
            self.num_workers,
        )
        print(f"\napproximate number of steps for val is {ceil(n_samples / self.hyperparams.test_batch_size)}")
        return dataloader

    def test_dataloader(self) -> DataLoader:
        dataloader, n_samples = create_dataloader(
            self.hyperparams.test_data_path,
            self.hyperparams.max_context,
            False,
            False,
            self.hyperparams.test_batch_size,
            self.num_workers,
        )
        print(f"\napproximate number of steps for test is {ceil(n_samples / self.hyperparams.test_batch_size)}")
        return dataloader

    # ===== STEP =====

    def transfer_batch_to_device(self, batch: PathContextBatch, device: torch.device) -> PathContextBatch:
        # Dict str -> torch.Tensor [seq length; batch size * n_context]
        for k in batch.context:
            batch.context[k] = batch.context[k].to(self.device)
        # [seq length; batch size]
        batch.labels = batch.labels.to(self.device)
        return batch

    # ===== ON EPOCH END =====

    def training_epoch_end(self, outputs: List[Dict]) -> Dict:
        return self._general_epoch_end(outputs, "loss", "train")

    def validation_epoch_end(self, outputs: List[Dict]) -> Dict:
        return self._general_epoch_end(outputs, "val_loss", "val")

    def test_epoch_end(self, outputs: List[Dict]) -> Dict:
        return self._general_epoch_end(outputs, "test_loss", "test")
