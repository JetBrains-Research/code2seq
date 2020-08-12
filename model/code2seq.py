from typing import Dict, List

import torch
import torch.nn.functional as F

from configs import Code2SeqConfig
from dataset import Vocabulary, PathContextBatch
from model.modules import PathEncoder, PathDecoder
from utils.common import PAD, SOS, UNK, EOS
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

    def _general_epoch_end(self, outputs: List[Dict], loss_key: str, group: str) -> Dict:
        with torch.no_grad():
            logs = {f"{group}/loss": torch.stack([out[loss_key] for out in outputs]).mean()}
            logs.update(
                SubtokenStatistic.union_statistics([out["statistic"] for out in outputs]).calculate_metrics(group)
            )
        progress_bar = {k: v for k, v in logs.items() if k in [f"{group}/loss", f"{group}/f1"]}
        return {f"{group}_loss": logs[f"{group}/loss"], "log": logs, "progress_bar": progress_bar}

    def training_step(self, batch: PathContextBatch, batch_idx: int) -> Dict:
        # [seq length; batch size; vocab size]
        logits = self(batch.context, batch.contexts_per_label, batch.labels.shape[0], batch.labels)
        loss = self._calculate_loss(logits, batch.labels)
        log = {"train/loss": loss}
        with torch.no_grad():
            statistic = SubtokenStatistic().calculate_statistic(
                batch.labels, logits.argmax(-1), [self.vocab.label_to_id[t] for t in [EOS, PAD, UNK]],
            )

        log.update(statistic.calculate_metrics(group="train"))
        progress_bar = {"train/f1": log["train/f1"]}

        return {"loss": loss, "log": log, "progress_bar": progress_bar, "statistic": statistic}

    def validation_step(self, batch: PathContextBatch, batch_idx: int) -> Dict:
        # [seq length; batch size; vocab size]
        logits = self(batch.context, batch.contexts_per_label, batch.labels.shape[0])
        loss = self._calculate_loss(logits, batch.labels)
        with torch.no_grad():
            statistic = SubtokenStatistic().calculate_statistic(
                batch.labels, logits.argmax(-1), [self.vocab.label_to_id[t] for t in [SOS, PAD, UNK]],
            )

        return {"val_loss": loss, "statistic": statistic}

    def test_step(self, batch: PathContextBatch, batch_idx: int) -> Dict:
        result = self.validation_step(batch, batch_idx)
        result["test_loss"] = result["val_loss"]
        del result["val_loss"]
        return result
