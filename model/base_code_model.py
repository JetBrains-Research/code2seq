from abc import ABCMeta, abstractmethod
from math import ceil
from typing import Tuple, Dict, List, Union

import torch
from pytorch_lightning.core.lightning import LightningModule
from torch.optim import Adam, Optimizer, SGD
from torch.optim.lr_scheduler import ExponentialLR, _LRScheduler
from torch.utils.data import DataLoader

from configs import ModelHyperparameters
from dataset import Vocabulary, create_dataloader, PathContextBatch
from utils.metrics import SubtokenStatistic, ClassificationStatistic

StatisticType = Union[SubtokenStatistic, ClassificationStatistic]


class BaseCodeModel(LightningModule, metaclass=ABCMeta):
    def __init__(
        self, config: ModelHyperparameters, vocab: Vocabulary, num_workers: int = 0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.vocab = vocab
        self.num_workers = num_workers

    def forward(
        self,
        samples: Dict[str, torch.Tensor],
        paths_for_label: List[int],
        output_length: int = None,
        target_sequence: torch.Tensor = None,
    ) -> torch.Tensor:
        return self.decoder(self.encoder(samples), paths_for_label, output_length, target_sequence)

    @abstractmethod
    def _calculate_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Calculate loss"""
        pass

    @abstractmethod
    def _compute_metrics(self, logits: torch.Tensor, labels: torch.Tensor) -> StatisticType:
        pass

    @abstractmethod
    def _general_epoch_end(self, outputs: List[Dict], loss_key: str, group: str) -> Dict:
        pass

    @abstractmethod
    def _get_progress_bar(self, log: Dict, group: str) -> Dict[str, float]:
        pass

    # ===== OPTIMIZERS =====

    def configure_optimizers(self) -> Tuple[List[Optimizer], List[_LRScheduler]]:
        if self.config.optimizer == "Momentum":
            # using the same momentum value as in original realization by Alon
            optimizer = SGD(
                self.parameters(),
                self.config.learning_rate,
                momentum=0.95,
                nesterov=self.config.nesterov,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer == "Adam":
            optimizer = Adam(self.parameters(), self.config.learning_rate, weight_decay=self.config.weight_decay)
        else:
            raise ValueError(f"Unknown optimizer name: {self.config.optimizer}, try one of: Adam, Momentum")
        scheduler = ExponentialLR(optimizer, self.config.decay_gamma)
        return [optimizer], [scheduler]

    # ===== DATALOADERS BLOCK =====

    def train_dataloader(self) -> DataLoader:
        dataloader, n_samples = create_dataloader(
            self.config.train_data_path,
            self.config.max_context,
            self.config.random_context,
            self.config.shuffle_data,
            self.config.batch_size,
            self.num_workers,
        )
        print(f"approximate number of steps for train is {ceil(n_samples / self.config.batch_size)}")
        return dataloader

    def val_dataloader(self) -> DataLoader:
        dataloader, n_samples = create_dataloader(
            self.config.val_data_path,
            self.config.max_context,
            False,
            False,
            self.config.test_batch_size,
            self.num_workers,
        )
        print(f"approximate number of steps for val is {ceil(n_samples / self.config.test_batch_size)}")
        return dataloader

    def test_dataloader(self) -> DataLoader:
        dataloader, n_samples = create_dataloader(
            self.config.test_data_path,
            self.config.max_context,
            False,
            False,
            self.config.test_batch_size,
            self.num_workers,
        )
        print(f"approximate number of steps for test is {ceil(n_samples / self.config.test_batch_size)}")
        return dataloader

    # ===== STEP =====

    def training_step(self, batch: PathContextBatch, batch_idx: int) -> Dict:
        # Dict str -> torch.Tensor [seq length; batch size * n_context]
        context = batch.context
        for k in context:
            context[k] = context[k].to(self.device)
        # [seq length; batch size]
        labels = batch.labels.to(self.device)

        # [seq length; batch size; vocab size]
        logits = self(context, batch.contexts_per_label, labels.shape[0], labels)
        loss = self._calculate_loss(logits, labels)
        log = {"train/loss": loss}
        with torch.no_grad():
            statistic = self._compute_metrics(logits, labels)

        log.update(statistic.calculate_metrics(group="train"))
        progress_bar = self._get_progress_bar(log, "train")

        return {"loss": loss, "log": log, "progress_bar": progress_bar, "statistic": statistic}

    def validation_step(self, batch: PathContextBatch, batch_idx: int) -> Dict:
        # Dict str -> torch.Tensor [seq length; batch size * n_context]
        context = batch.context
        for k in context:
            context[k] = context[k].to(self.device)
        # [seq length; batch size]
        labels = batch.labels.to(self.device)

        # [seq length; batch size; vocab size]
        logits = self(context, batch.contexts_per_label, labels.shape[0])
        loss = self._calculate_loss(logits, labels)
        with torch.no_grad():
            statistic = self._compute_metrics(logits, labels)

        return {"val_loss": loss, "statistic": statistic}

    def test_step(self, batch: PathContextBatch, batch_idx: int) -> Dict:
        result = self.validation_step(batch, batch_idx)
        result["test_loss"] = result["val_loss"]
        del result["val_loss"]
        return result

    # ===== ON EPOCH END =====

    def training_epoch_end(self, outputs: List[Dict]) -> Dict:
        return self._general_epoch_end(outputs, "loss", "train")

    def validation_epoch_end(self, outputs: List[Dict]) -> Dict:
        return self._general_epoch_end(outputs, "val_loss", "val")

    def test_epoch_end(self, outputs: List[Dict]) -> Dict:
        return self._general_epoch_end(outputs, "test_loss", "test")
