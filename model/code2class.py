from typing import Dict, List

import torch
import torch.nn.functional as F

from model.modules import PathEncoder, PathClassifier
from utils.common import PAD
from utils.metrics import ClassificationStatistic
from .base_code_model import BaseCodeModel, EncoderConfigType, DecoderConfigType, StatisticType


class Code2Class(BaseCodeModel):
    def _init_models(self, encoder_config: EncoderConfigType, decoder_config: DecoderConfigType):
        self.encoder = PathEncoder(
            encoder_config,
            decoder_config.classifier_input_size,
            len(self.vocab.token_to_id),
            self.vocab.token_to_id[PAD],
            len(self.vocab.type_to_id),
            self.vocab.type_to_id[PAD],
        )
        self.decoder = PathClassifier(decoder_config, len(self.vocab.label_to_id))

    def _calculate_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """ Calculate cross entropy loss

        :param logits: [batch size; vocab size]
        :param labels: [1; batch size]
        :return: [1]
        """
        # [1, batch size]
        labels = labels.squeeze(0)
        loss = F.cross_entropy(logits, labels).mean()
        return loss

    def _compute_metrics(self, logits: torch.Tensor, labels: torch.Tensor) -> StatisticType:
        classification_statistic = ClassificationStatistic(len(self.vocab.type_to_id)).calculate_statistic(
            labels.detach().squeeze(0), logits.detach().argmax(-1),
        )
        return classification_statistic

    def _get_progress_bar(self, log: Dict, group: str) -> Dict:
        return {f"{group}/acc": log[f"{group}/accuracy"]}

    def _general_epoch_end(self, outputs: List[Dict], loss_key: str, group: str) -> Dict:
        logs = {f"{group}/loss": torch.stack([out[loss_key] for out in outputs]).mean()}
        logs.update(
            ClassificationStatistic(len(self.vocab.type_to_id))
            .union_statistics([out["statistic"] for out in outputs])
            .calculate_metrics(group)
        )
        progress_bar = {k: v for k, v in logs.items() if k in [f"{group}/loss", f"{group}/acc"]}

        return {"val_loss": logs[f"{group}/loss"], "log": logs, "progress_bar": progress_bar}
