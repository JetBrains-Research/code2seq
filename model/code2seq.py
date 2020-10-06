from typing import Dict, List

import torch
import torch.nn.functional as F

from configs import Code2SeqConfig
from dataset import Vocabulary, PathContextBatch
from model.modules import PathEncoder, PathDecoder
from utils.common import PAD, SOS, UNK
from utils.metrics import SubtokenStatistic
from .base_code_model import BaseCodeModel


class Code2Seq(BaseCodeModel):
    def __init__(self, config: Code2SeqConfig, vocab: Vocabulary, num_workers: int):
        super().__init__(config.hyperparams, vocab, num_workers)
        self.save_hyperparameters()
        if SOS not in vocab.label_to_id:
            vocab.label_to_id[SOS] = len(vocab.label_to_id)
        encoder_config = config.encoder_config
        decoder_config = config.decoder_config
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

    def forward(
        self,
        samples: Dict[str, torch.Tensor],
        paths_for_label: List[int],
        output_length: int,
        target_sequence: torch.Tensor = None,
    ) -> torch.Tensor:
        return self.decoder(self.encoder(samples), paths_for_label, output_length, target_sequence)

    def _calculate_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Calculate cross entropy with ignoring PAD index

        :param logits: [seq length; batch size; vocab size]
        :param labels: [seq length; batch size]
        :return: [1]
        """
        batch_size = labels.shape[-1]
        # [batch size; vocab size; seq length]
        _logits = logits.permute(1, 2, 0)
        # [batch size; seq length]
        _labels = labels.permute(1, 0)
        # [batch size; seq length]
        loss = F.cross_entropy(_logits, _labels, reduction="none")
        # [batch size; seq length]
        mask = _labels != self.vocab.label_to_id[PAD]
        # [batch size; seq length]
        loss = loss * mask
        # [1]
        loss = loss.sum() / batch_size
        return loss

    def _general_epoch_end(self, outputs: List[Dict], loss_key: str, group: str) -> Dict:
        with torch.no_grad():
            logs = {f"{group}/loss": torch.stack([out[loss_key] for out in outputs]).mean()}
            logs.update(
                SubtokenStatistic.union_statistics([out["statistic"] for out in outputs]).calculate_metrics(group)
            )
        progress_bar = {k: v for k, v in logs.items() if k in [f"{group}/loss", f"{group}/f1"]}
        return {f"{group}_loss": logs[f"{group}/loss"], "log": logs, "progress_bar": progress_bar}

    def _calculate_metric(self, logits: torch.Tensor, labels: torch.Tensor) -> SubtokenStatistic:
        with torch.no_grad():
            # [seq length; batch size]
            prediction = logits.argmax(-1)
            mask_max_value, mask_max_indices = torch.max(prediction == self.vocab.label_to_id[PAD], dim=0)
            mask_max_indices[~mask_max_value] = prediction.shape[0]
            mask = torch.arange(prediction.shape[0], device=self.device).view(-1, 1) >= mask_max_indices
            prediction[mask] = self.vocab.label_to_id[PAD]
            statistic = SubtokenStatistic().calculate_statistic(
                labels, prediction, [self.vocab.label_to_id[t] for t in [PAD, UNK]],
            )
        return statistic

    def training_step(self, batch: PathContextBatch, batch_idx: int) -> Dict:
        logits = self(batch.context, batch.contexts_per_label, batch.labels.shape[0], batch.labels)
        loss = self._calculate_loss(logits, batch.labels)
        log = {"train/loss": loss}
        statistic = self._calculate_metric(logits, batch.labels)

        log.update(statistic.calculate_metrics(group="train"))
        progress_bar = {"train/f1": log["train/f1"]}

        return {"loss": loss, "log": log, "progress_bar": progress_bar, "statistic": statistic}

    def validation_step(self, batch: PathContextBatch, batch_idx: int) -> Dict:
        # [seq length; batch size; vocab size]
        logits = self(batch.context, batch.contexts_per_label, batch.labels.shape[0])
        loss = self._calculate_loss(logits, batch.labels)
        statistic = self._calculate_metric(logits, batch.labels)
        return {"val_loss": loss, "statistic": statistic}

    def test_step(self, batch: PathContextBatch, batch_idx: int) -> Dict:
        result = self.validation_step(batch, batch_idx)
        result["test_loss"] = result.pop("val_loss")
        return result
