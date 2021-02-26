from typing import Dict, List, Union, Tuple

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from code2seq.dataset import PathContextBatch
from code2seq.model.modules import PathEncoder, PathDecoder
from code2seq.utils.metrics import PredictionStatistic
from code2seq.utils.training import configure_optimizers_alon
from code2seq.utils.vocabulary import Vocabulary, SOS, PAD, UNK, EOS


class Code2Seq(LightningModule):
    def __init__(self, config: DictConfig, vocabulary: Vocabulary):
        super().__init__()
        self._config = config
        self._vocabulary = vocabulary
        self.save_hyperparameters()

        self._metric_skip_tokens = [
            vocabulary.label_to_id[i] for i in [PAD, UNK, EOS, SOS] if i in vocabulary.label_to_id
        ]
        self._label_pad_id = vocabulary.label_to_id[PAD]

        if SOS not in vocabulary.label_to_id:
            raise ValueError(f"Can't find SOS token in label to id vocabulary")
        self.encoder = self._get_encoder()
        self.decoder = self._get_decoder()

    @property
    def config(self) -> DictConfig:
        return self._config

    @property
    def vocabulary(self) -> Vocabulary:
        return self._vocabulary

    # ========== Create seq2seq modules ==========

    def _get_encoder(self) -> PathEncoder:
        return PathEncoder(
            self._config.encoder,
            self._config.decoder.decoder_size,
            len(self._vocabulary.token_to_id),
            self._vocabulary.token_to_id[PAD],
            len(self._vocabulary.node_to_id),
            self._vocabulary.node_to_id[PAD],
        )

    def _get_decoder(self) -> PathDecoder:
        return PathDecoder(
            self._config.decoder,
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

    # ========== Model step ==========

    def training_step(self, batch: PathContextBatch, batch_idx: int) -> Dict:  # type: ignore
        # [seq length; batch size; vocab size]
        logits = self(batch.contexts, batch.contexts_per_label, batch.labels.shape[0], batch.labels)
        loss = self._calculate_loss(logits, batch.labels)
        prediction = logits.argmax(-1)

        statistic = PredictionStatistic(True, self._label_pad_id, self._metric_skip_tokens)
        batch_metric = statistic.update_statistic(batch.labels, prediction)

        log: Dict[str, Union[float, torch.Tensor]] = {"train/loss": loss}
        for key, value in batch_metric.items():
            log[f"train/{key}"] = value
        self.log_dict(log)
        self.log("f1", batch_metric["f1"], prog_bar=True, logger=False)

        return {"loss": loss, "statistic": statistic}

    def validation_step(self, batch: PathContextBatch, batch_idx: int) -> Dict:  # type: ignore
        # [seq length; batch size; vocab size]
        logits = self(batch.contexts, batch.contexts_per_label, batch.labels.shape[0])
        loss = self._calculate_loss(logits, batch.labels)
        prediction = logits.argmax(-1)

        statistic = PredictionStatistic(True, self._label_pad_id, self._metric_skip_tokens)
        statistic.update_statistic(batch.labels, prediction)

        return {"loss": loss, "statistic": statistic}

    def test_step(self, batch: PathContextBatch, batch_idx: int) -> Dict:  # type: ignore
        return self.validation_step(batch, batch_idx)

    # ========== On epoch end ==========

    def _shared_epoch_end(self, outputs: List[Dict], group: str):
        with torch.no_grad():
            mean_loss = torch.stack([out["loss"] for out in outputs]).mean().item()
            statistic = PredictionStatistic.create_from_list([out["statistic"] for out in outputs])
            epoch_metrics = statistic.get_metric()
            log: Dict[str, Union[float, torch.Tensor]] = {f"{group}/loss": mean_loss}
            for key, value in epoch_metrics.items():
                log[f"{group}/{key}"] = value
            self.log_dict(log)
            self.log(f"{group}_loss", mean_loss)

    def training_epoch_end(self, outputs: List[Dict]):
        self._shared_epoch_end(outputs, "train")

    def validation_epoch_end(self, outputs: List[Dict]):
        self._shared_epoch_end(outputs, "val")

    def test_epoch_end(self, outputs: List[Dict]):
        self._shared_epoch_end(outputs, "test")
