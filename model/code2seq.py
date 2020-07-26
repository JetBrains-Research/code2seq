from typing import Dict, List

import torch
import torch.nn.functional as F

from model.modules import PathEncoder, PathDecoder
from utils.common import PAD, SOS
from utils.metrics import SubtokenStatistic
from .base_code_model import BaseCodeModel, EncoderConfigType, DecoderConfigType, StatisticType


class Code2Seq(BaseCodeModel):
    def _init_models(self, encoder_config: EncoderConfigType, decoder_config: DecoderConfigType):
        self.encoder = PathEncoder(
            encoder_config,
            decoder_config.decoder_size,
            len(self.vocab.token_to_id),
            self.vocab.token_to_id[PAD],
            len(self.vocab.type_to_id),
            self.vocab.type_to_id[PAD],
        )
        self.decoder = PathDecoder(
            decoder_config, len(self.vocab.label_to_id), self.vocab.label_to_id[SOS], self.vocab.label_to_id[PAD]
        )

    def _calculate_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """ Calculate cross entropy loss with removing SOS tokens and masking PAD values.
        Adaptation of tf.nn.sparse_softmax_cross_entropy_with_logits from original implementation.

        :param logits: [seq length; batch size; vocab size]
        :param labels: [seq length; batch size]
        :return: [1]
        """
        # remove SOS token
        # [batch size; vocab size; seq length]
        logits = logits[1:].permute(1, 2, 0)
        # [batch size; seq length]
        labels = labels[1:].transpose(0, 1)

        loss = F.cross_entropy(logits, labels, reduction="none")

        with torch.no_grad():
            label_mask = labels != self.vocab.label_to_id[PAD]
        loss = (loss * label_mask).sum() / labels.shape[0]
        return loss

    def _compute_metrics(self, logits: torch.Tensor, labels: torch.Tensor) -> StatisticType:
        classification_statistic = SubtokenStatistic().calculate_statistic(labels.detach(), logits.detach().argmax(-1))
        return classification_statistic

    def _get_progress_bar(self, log: Dict, group: str) -> Dict:
        return {f"{group}/f1": log[f"{group}/f1"]}

    def _general_epoch_end(self, outputs: List[Dict], loss_key: str, group: str) -> Dict:
        logs = {f"{group}/loss": torch.stack([out[loss_key] for out in outputs]).mean()}
        logs.update(SubtokenStatistic.union_statistics([out["statistic"] for out in outputs]).calculate_metrics(group))
        progress_bar = {k: v for k, v in logs.items() if k in [f"{group}/loss", f"{group}/f1"]}
        return {"val_loss": logs[f"{group}/loss"], "log": logs, "progress_bar": progress_bar}
