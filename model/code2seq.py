from math import ceil
from typing import Tuple, Dict, List

import torch
import torch.nn.functional as F
from pytorch_lightning.core.lightning import LightningModule
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader

from configs import Code2SeqConfig
from dataset import Vocabulary, create_dataloader, PathContextBatch
from model.modules import PathEncoder, PathDecoder
from utils.common import PAD, SOS, EOS, UNK
from utils.metrics import SubtokenStatistic


class Code2Seq(LightningModule):
    def __init__(self, config: Code2SeqConfig, vocab: Vocabulary):
        super().__init__()
        self.config = config
        self.vocab = vocab

        encoder_config = self.config.encoder
        decoder_config = self.config.decoder
        self.encoder = PathEncoder(
            encoder_config,
            decoder_config.decoder_size,
            len(vocab.token_to_id),
            vocab.token_to_id[PAD],
            len(vocab.type_to_id),
            vocab.type_to_id[PAD],
        )
        self.decoder = PathDecoder(
            decoder_config, len(vocab.label_to_id), vocab.label_to_id[SOS], vocab.label_to_id[PAD]
        )

    def forward(self, samples: Dict[str, torch.Tensor], paths_for_label: List[int], output_length: int) -> torch.Tensor:
        return self.decoder(self.encoder(samples), paths_for_label, output_length)

    def configure_optimizers(self) -> Optimizer:
        return Adam(self.parameters(), self.config.learning_rate)

    def _calculate_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """ Calculate cross entropy loss.

        :param logits: [seq length; batch size; vocab size]
        :param labels: [seq length; batch size]
        :return: [1]
        """
        # [(seq length - 1) * batch size; vocab size]
        logits = logits[1:].view(-1, logits.shape[-1])
        # [(seq length - 1) * batch size]
        labels = labels[1:].view(-1)
        loss = F.cross_entropy(
            logits.view(-1, logits.shape[-1]), labels.view(-1), ignore_index=self.vocab.label_to_id[PAD]
        )
        return loss

    def _general_forward_step(self, batch: PathContextBatch) -> Tuple[torch.Tensor, SubtokenStatistic]:
        # Dict str -> torch.Tensor [seq length; batch size * n_context]
        context = batch.context
        for k in context:
            context[k] = context[k].to(self.device)
        # [seq length; batch size]
        labels = batch.labels.to(self.device)

        # [seq length; batch size; vocab size]
        logits = self(context, batch.contexts_per_label, labels.shape[0])
        loss = self._calculate_loss(logits, labels)

        subtoken_statistic = SubtokenStatistic.calculate_statistic(
            labels, logits.argmax(-1), [self.vocab.label_to_id[t] for t in [SOS, EOS, PAD, UNK]]
        )
        return loss, subtoken_statistic

    def _general_epoch_end(self, outputs: List[Dict], loss_key: str, group: str) -> Dict:
        logs = {f"{group}/loss": torch.stack([out[loss_key] for out in outputs]).mean()}
        logs.update(
            SubtokenStatistic.union_statistics([out["subtoken_statistic"] for out in outputs]).calculate_metrics(group)
        )
        progress_bar = {k: v for k, v in logs.items() if k in [f"{group}/loss", f"{group}/f1"]}
        return {"val_loss": logs[f"{group}/loss"], "log": logs, "progress_bar": progress_bar}

    # ===== TRAIN BLOCK =====

    def train_dataloader(self) -> DataLoader:
        dataloader, n_samples = create_dataloader(
            self.config.train_data_path,
            self.config.max_context,
            self.config.random_context,
            self.config.shuffle_data,
            self.config.batch_size,
            self.config.num_workers,
        )
        print(f"approximate number of steps for train is {ceil(n_samples / self.config.batch_size)}")
        return dataloader

    def training_step(self, batch: PathContextBatch, batch_idx: int) -> Dict:
        loss, subtoken_statistic = self._general_forward_step(batch)
        log = {"train/loss": loss}
        log.update(subtoken_statistic.calculate_metrics(group="train"))
        progress_bar = {"train/f1": log["train/f1"]}
        return {"loss": loss, "log": log, "progress_bar": progress_bar, "subtoken_statistic": subtoken_statistic}

    def training_epoch_end(self, outputs: List[Dict]) -> Dict:
        return self._general_epoch_end(outputs, "loss", "train")

    # ===== VALIDATION BLOCK =====

    def val_dataloader(self) -> DataLoader:
        dataloader, n_samples = create_dataloader(
            self.config.val_data_path,
            self.config.max_context,
            False,
            False,
            self.config.test_batch_size,
            self.config.num_workers,
        )
        print(f"approximate number of steps for val is {ceil(n_samples / self.config.test_batch_size)}")
        return dataloader

    def validation_step(self, batch: PathContextBatch, batch_idx: int) -> Dict:
        loss, subtoken_statistic = self._general_forward_step(batch)
        return {"val_loss": loss, "subtoken_statistic": subtoken_statistic}

    def validation_epoch_end(self, outputs: List[Dict]) -> Dict:
        return self._general_epoch_end(outputs, "val_loss", "val")

    # ===== TEST BLOCK =====

    def test_dataloader(self) -> DataLoader:
        dataloader, n_samples = create_dataloader(
            self.config.test_data_path,
            self.config.max_context,
            False,
            False,
            self.config.test_batch_size,
            self.config.num_workers,
        )
        print(f"approximate number of steps for test is {ceil(n_samples / self.config.test_batch_size)}")
        return dataloader

    def test_step(self, batch: Tuple[Dict[str, torch.Tensor], torch.Tensor], batch_idx: int) -> Dict:
        pass

    def test_epoch_end(self, outputs: List[Dict]) -> Dict:
        return self._general_epoch_end(outputs, "test_loss", "test")
