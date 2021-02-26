from typing import Dict, List, Tuple, Union

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from pytorch_lightning.metrics.functional import confusion_matrix
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from code2seq.dataset import PathContextBatch
from code2seq.model.modules import PathEncoder, PathClassifier
from code2seq.utils.training import configure_optimizers_alon
from code2seq.utils.vocabulary import Vocabulary, PAD


class Code2Class(LightningModule):
    def __init__(self, config: DictConfig, vocabulary: Vocabulary):
        super().__init__()
        self._config = config
        self.save_hyperparameters()
        self.encoder = PathEncoder(
            self._config.encoder,
            self._config.classifier.classifier_input_size,
            len(vocabulary.token_to_id),
            vocabulary.token_to_id[PAD],
            len(vocabulary.node_to_id),
            vocabulary.node_to_id[PAD],
        )
        self.num_classes = len(vocabulary.label_to_id)
        self.classifier = PathClassifier(self._config.classifier, self.num_classes)

    def configure_optimizers(self) -> Tuple[List[Optimizer], List[_LRScheduler]]:
        return configure_optimizers_alon(self._config.hyper_parameters, self.parameters())

    def forward(self, samples: Dict[str, torch.Tensor], paths_for_label: List[int]) -> torch.Tensor:  # type: ignore
        return self.classifier(self.encoder(samples), paths_for_label)

    # ========== MODEL STEP ==========

    def training_step(self, batch: PathContextBatch, batch_idx: int) -> Dict:  # type: ignore
        # [batch size; num_classes]
        logits = self(batch.contexts, batch.contexts_per_label)
        loss = F.cross_entropy(logits, batch.labels.squeeze(0))
        log = {"train/loss": loss}
        with torch.no_grad():
            conf_matrix = confusion_matrix(logits.argmax(-1), batch.labels.squeeze(0), self.num_classes)
            log["train/accuracy"] = conf_matrix.trace() / conf_matrix.sum()
        self.log_dict(log)

        return {"loss": loss, "confusion_matrix": conf_matrix}

    def validation_step(self, batch: PathContextBatch, batch_idx: int) -> Dict:  # type: ignore
        # [batch size; num_classes]
        logits = self(batch.contexts, batch.contexts_per_label)
        loss = F.cross_entropy(logits, batch.labels.squeeze(0))
        with torch.no_grad():
            conf_matrix = confusion_matrix(logits.argmax(-1), batch.labels.squeeze(0), self.num_classes)

        return {"loss": loss, "confusion_matrix": conf_matrix}

    def test_step(self, batch: PathContextBatch, batch_idx: int) -> Dict:  # type: ignore
        return self.validation_step(batch, batch_idx)

    # ========== ON EPOCH END ==========

    def _general_epoch_end(self, outputs: List[Dict], group: str):
        with torch.no_grad():
            mean_loss = torch.stack([out["loss"] for out in outputs]).mean().item()
            log: Dict[str, Union[float, torch.Tensor]] = {f"{group}/loss": mean_loss}
            accumulated_conf_matrix = torch.zeros(
                self.num_classes, self.num_classes, requires_grad=False, device=self.device
            )
            for out in outputs:
                _conf_matrix = out["confusion_matrix"]
                max_class_index, _ = _conf_matrix.shape
                accumulated_conf_matrix[:max_class_index, :max_class_index] += _conf_matrix
            log[f"{group}/accuracy"] = (accumulated_conf_matrix.trace() / accumulated_conf_matrix.sum()).item()
            self.log_dict(log)
            self.log(f"{group}_loss", mean_loss)

    def training_epoch_end(self, outputs: List[Dict]):
        self._general_epoch_end(outputs, "train")

    def validation_epoch_end(self, outputs: List[Dict]):
        self._general_epoch_end(outputs, "val")

    def test_epoch_end(self, outputs: List[Dict]):
        self._general_epoch_end(outputs, "test")
