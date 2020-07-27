from typing import Dict, List

import torch
import torch.nn.functional as F

from configs import ModelHyperparameters, EncoderConfig, ClassifierConfig
from dataset import Vocabulary, PathContextBatch
from model.modules import PathEncoder, PathClassifier
from utils.common import PAD
from utils.metrics import ClassificationStatistic
from .base_code_model import BaseCodeModel


class Code2Class(BaseCodeModel):
    def __init__(
        self,
        hyperparams: ModelHyperparameters,
        vocab: Vocabulary,
        num_workers: int,
        encoder_config: EncoderConfig,
        decoder_config: ClassifierConfig,
    ):
        super().__init__(hyperparams, vocab, num_workers)
        self.encoder = PathEncoder(
            encoder_config,
            decoder_config.classifier_input_size,
            len(self.vocab.token_to_id),
            self.vocab.token_to_id[PAD],
            len(self.vocab.type_to_id),
            self.vocab.type_to_id[PAD],
        )
        self.decoder = PathClassifier(decoder_config, len(self.vocab.label_to_id))

    def forward(self, samples: Dict[str, torch.Tensor], paths_for_label: List[int],) -> torch.Tensor:
        return self.decoder(self.encoder(samples), paths_for_label)

    def _general_epoch_end(self, outputs: List[Dict], loss_key: str, group: str) -> Dict:
        logs = {f"{group}/loss": torch.stack([out[loss_key] for out in outputs]).mean()}
        logs.update(
            ClassificationStatistic(len(self.vocab.label_to_id))
            .union_statistics([out["statistic"] for out in outputs])
            .calculate_metrics(group)
        )
        progress_bar = {k: v for k, v in logs.items() if k in [f"{group}/loss", f"{group}/accuracy"]}

        return {"val_loss": logs[f"{group}/loss"], "log": logs, "progress_bar": progress_bar}

    def training_step(self, batch: PathContextBatch, batch_idx: int) -> Dict:
        # [seq length; batch size; vocab size]
        logits = self(batch.context, batch.contexts_per_label, batch.labels.shape[0], batch.labels)
        loss = torch.nn.CrossEntropyLoss(logits, batch.labels)
        log = {"train/loss": loss}
        with torch.no_grad():
            statistic = ClassificationStatistic(len(self.vocab.label_to_id)).calculate_statistic(
                batch.labels.detach().squeeze(0), logits.detach().argmax(-1),
            )

        log.update(statistic.calculate_metrics(group="train"))
        progress_bar = {f"train/accuracy": log[f"train/accuracy"]}

        return {"loss": loss, "log": log, "progress_bar": progress_bar, "statistic": statistic}

    def validation_step(self, batch: PathContextBatch, batch_idx: int) -> Dict:
        # [seq length; batch size; vocab size]
        logits = self(batch.context, batch.contexts_per_label, batch.labels.shape[0])
        loss = torch.nn.CrossEntropyLoss(logits, batch.labels)
        with torch.no_grad():
            classification_statistic = ClassificationStatistic(len(self.vocab.label_to_id)).calculate_statistic(
                batch.labels.detach().squeeze(0), logits.detach().argmax(-1),
            )

        return {"val_loss": loss, "classification_statistic": classification_statistic}

    def test_step(self, batch: PathContextBatch, batch_idx: int) -> Dict:
        result = self.validation_step(batch, batch_idx)
        result["test_loss"] = result["val_loss"]
        del result["val_loss"]
        return result