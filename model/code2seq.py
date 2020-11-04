from typing import Dict, List, Union, Tuple

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from configs import Code2SeqConfig
from dataset import PathContextBatch
from model.modules import PathEncoder, PathDecoder
from utils.common import PAD, SOS, UNK, EOS
from utils.metrics import SubtokenStatistic
from utils.training import configure_optimizers_alon
from utils.vocabulary import Vocabulary


class Code2Seq(LightningModule):
    def __init__(self, config: Code2SeqConfig, vocabulary: Vocabulary):
        super().__init__()
        self._config = config
        self._vocabulary = vocabulary
        self.save_hyperparameters()

        if SOS not in vocabulary.label_to_id:
            raise ValueError(f"Can't find SOS token in label to id vocabulary")
        self.encoder = self._get_encoder()
        self.decoder = self._get_decoder()

    @property
    def config(self) -> Code2SeqConfig:
        return self._config

    @property
    def vocabulary(self) -> Vocabulary:
        return self._vocabulary

    # ========== Create seq2seq modules ==========

    def _get_encoder(self) -> PathEncoder:
        return PathEncoder(
            self._config.encoder_config,
            self._config.decoder_config.decoder_size,
            len(self._vocabulary.token_to_id),
            self._vocabulary.token_to_id[PAD],
            len(self._vocabulary.node_to_id),
            self._vocabulary.node_to_id[PAD],
        )

    def _get_decoder(self) -> PathDecoder:
        return PathDecoder(
            self._config.decoder_config,
            len(self._vocabulary.label_to_id),
            self._vocabulary.label_to_id[SOS],
            self._vocabulary.label_to_id[PAD],
        )

    # ========== Main PyTorch-Lightning hooks ==========

    def configure_optimizers(self) -> Tuple[List[Optimizer], List[_LRScheduler]]:
        return configure_optimizers_alon(self._config.hyper_parameters, self.parameters())

    def forward(  # type: ignore
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
        mask = _labels != self._vocabulary.label_to_id[PAD]
        # [batch size; seq length]
        loss = loss * mask
        # [1]
        loss = loss.sum() / batch_size
        return loss

    def _calculate_metric(self, logits: torch.Tensor, labels: torch.Tensor) -> SubtokenStatistic:
        with torch.no_grad():
            # [seq length; batch size]
            prediction = logits.argmax(-1)
            mask_max_value, mask_max_indices = torch.max(prediction == self._vocabulary.label_to_id[PAD], dim=0)
            mask_max_indices[~mask_max_value] = prediction.shape[0]
            mask = torch.arange(prediction.shape[0], device=self.device).view(-1, 1) >= mask_max_indices
            prediction[mask] = self._vocabulary.label_to_id[PAD]
            statistic = SubtokenStatistic().calculate_statistic(
                labels, prediction, [self._vocabulary.label_to_id[t] for t in [PAD, UNK, EOS, SOS]],
            )
        return statistic

    # ========== Model step ==========

    def training_step(self, batch: PathContextBatch, batch_idx: int) -> Dict:  # type: ignore
        logits = self(batch.contexts, batch.contexts_per_label, batch.labels.shape[0], batch.labels)
        loss = self._calculate_loss(logits, batch.labels)

        log: Dict[str, Union[float, torch.Tensor]] = {"train/loss": loss}
        statistic = self._calculate_metric(logits, batch.labels)
        log.update(statistic.calculate_metrics(group="train"))
        self.log_dict(log)

        return {"loss": loss, "statistic": statistic}

    def validation_step(self, batch: PathContextBatch, batch_idx: int) -> Dict:  # type: ignore
        # [seq length; batch size; vocab size]
        logits = self(batch.contexts, batch.contexts_per_label, batch.labels.shape[0])
        loss = self._calculate_loss(logits, batch.labels)
        statistic = self._calculate_metric(logits, batch.labels)
        return {"loss": loss, "statistic": statistic}

    def test_step(self, batch: PathContextBatch, batch_idx: int) -> Dict:  # type: ignore
        return self.validation_step(batch, batch_idx)

    # ========== On epoch end ==========

    def _general_epoch_end(self, outputs: List[Dict], group: str):
        with torch.no_grad():
            logs: Dict[str, Union[float, torch.Tensor]] = {
                f"{group}/loss": torch.stack([out["loss"] for out in outputs]).mean()
            }
            logs.update(
                SubtokenStatistic.union_statistics([out["statistic"] for out in outputs]).calculate_metrics(group)
            )
            self.log_dict(logs)

    def training_epoch_end(self, outputs: List[Dict]):
        self._general_epoch_end(outputs, "train")

    def validation_epoch_end(self, outputs: List[Dict]):
        self._general_epoch_end(outputs, "val")

    def test_epoch_end(self, outputs: List[Dict]):
        self._general_epoch_end(outputs, "test")
