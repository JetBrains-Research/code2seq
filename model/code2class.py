from typing import Dict, List

import torch
import torch.nn.functional as F

from configs import ModelHyperparameters, EncoderConfig, ClassifierConfig
from dataset import Vocabulary, PathContextBatch
from model.modules import PathEncoder, PathClassifier
from utils.common import PAD
from .base_code_model import BaseCodeModel


class Code2Class(BaseCodeModel):
    def __init__(
        self,
        hyperparams: ModelHyperparameters,
        vocab: Vocabulary,
        num_workers: int,
        encoder_config: EncoderConfig,
        classifier_config: ClassifierConfig,
    ):
        super().__init__(hyperparams, vocab, num_workers)
        self.encoder = PathEncoder(
            encoder_config,
            classifier_config.classifier_input_size,
            len(self.vocab.token_to_id),
            self.vocab.token_to_id[PAD],
            len(self.vocab.type_to_id),
            self.vocab.type_to_id[PAD],
        )
        self.classifier = PathClassifier(classifier_config, len(self.vocab.label_to_id))

    def forward(self, samples: Dict[str, torch.Tensor], paths_for_label: List[int]) -> torch.Tensor:
        return self.classifier(self.encoder(samples), paths_for_label)

    def _general_epoch_end(self, outputs: List[Dict], loss_key: str, group: str) -> Dict:
        logs = {f"{group}/loss": torch.stack([out[loss_key] for out in outputs]).mean()}
        # Computing accuracy throughout the epoch
        correct = sum([out[f"{group}/correct"] for out in outputs])
        total = sum([out[f"{group}/total"] for out in outputs])
        logs.update({f"{group}/accuracy": correct / total})

        progress_bar = {k: v for k, v in logs.items() if k in [f"{group}/loss", f"{group}/accuracy"]}

        return {"val_loss": logs[f"{group}/loss"], "log": logs, "progress_bar": progress_bar}

    @staticmethod
    def _count_correct(logits: torch.Tensor, labels: torch.Tensor) -> int:
        """ Calculate correctly predicted labels
        :param logits: [batch size; num classes]
        :param labels: [batch size]
        :return: [1]
        """
        indexes = logits.argmax(-1).detach()
        labels = labels.detach()
        return (indexes == labels).sum().item()

    def training_step(self, batch: PathContextBatch, batch_idx: int) -> Dict:
        # [batch size; num_classes]
        logits = self(batch.context, batch.contexts_per_label)
        loss = F.cross_entropy(logits, batch.labels.squeeze(0))
        with torch.no_grad():
            correct, total = self._count_correct(logits, batch.labels.squeeze(0)), logits.shape[0]
            log = {
                "train/loss": loss,
                "train/accuracy": correct / total,
                "train/total": total,
                "train/correct": correct,
            }
        progress_bar = {"train/accuracy": log["train/accuracy"]}
        return {"loss": loss, "log": log, "progress_bar": progress_bar, "train/total": total, "train/correct": correct}

    def validation_step(self, batch: PathContextBatch, batch_idx: int) -> Dict:
        # [batch size; num_classes]
        logits = self(batch.context, batch.contexts_per_label)
        loss = F.cross_entropy(logits, batch.labels.squeeze(0))
        with torch.no_grad():
            correct, total = self._count_correct(logits, batch.labels.squeeze(0)), logits.shape[0]

        return {"val_loss": loss, "val/total": total, "val/correct": correct}

    def test_step(self, batch: PathContextBatch, batch_idx: int) -> Dict:
        result = self.validation_step(batch, batch_idx)
        result["test_loss"] = result["val_loss"]
        del result["val_loss"]
        return result
