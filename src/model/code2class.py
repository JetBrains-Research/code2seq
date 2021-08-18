from typing import Dict, List, Tuple

import torch
from commode_utils.modules import Classifier
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torchmetrics import Metric, Accuracy, MetricCollection

from src.data.path_context import BatchedLabeledPathContext
from src.data.vocabulary import Vocabulary
from src.model.modules import PathEncoder
from src.utils.optimization import configure_optimizers_alon


class Code2Class(LightningModule):
    def __init__(self, model_config: DictConfig, optimizer_config: DictConfig, vocabulary: Vocabulary):
        super().__init__()
        self.save_hyperparameters()
        self._optim_config = optimizer_config

        self._encoder = PathEncoder(
            model_config,
            len(vocabulary.token_to_id),
            vocabulary.token_to_id[Vocabulary.PAD],
            len(vocabulary.node_to_id),
            vocabulary.node_to_id[Vocabulary.PAD],
        )

        self._classifier = Classifier(model_config, self._num_classes)

        metrics: Dict[str, Metric] = {
            f"{holdout}_acc": Accuracy(num_classes=len(vocabulary.label_to_id)) for holdout in ["train", "val", "test"]
        }
        self.__metrics = MetricCollection(metrics)

    def configure_optimizers(self) -> Tuple[List[Optimizer], List[_LRScheduler]]:
        return configure_optimizers_alon(self._config.hyper_parameters, self.parameters())

    def forward(  # type: ignore
        self,
        from_token: torch.Tensor,
        path_nodes: torch.Tensor,
        to_token: torch.Tensor,
        contexts_per_label: torch.Tensor,
    ) -> torch.Tensor:
        encoded_paths = self.__encoder(from_token, path_nodes, to_token)
        output_logits = self._classifier(encoded_paths, contexts_per_label)
        return output_logits

    # ========== MODEL STEP ==========

    def _shared_step(self, batch: BatchedLabeledPathContext, step: str) -> Dict:
        # [batch size; num_classes]
        logits = self(batch.from_token, batch.path_nodes, batch.to_token, batch.contexts_per_label)
        loss = torch.nn.functional.cross_entropy(logits, batch.labels.squeeze(0))

        with torch.no_grad():
            predictions = logits.argmax(-1)
            accuracy = self.__metrics[f"{step}_acc"](predictions, batch.labels)

        return {f"{step}/loss": loss, f"{step}/accuracy": accuracy}

    def training_step(self, batch: BatchedLabeledPathContext, batch_idx: int) -> Dict:  # type: ignore
        result = self._shared_step(batch, "train")
        self.log_dict(result, on_step=True, on_epoch=False)
        self.log("acc", result["train/accuracy"], prog_bar=True, logger=False)
        return result["train/loss"]

    def validation_step(self, batch: BatchedLabeledPathContext, batch_idx: int) -> Dict:  # type: ignore
        return self._shared_step(batch, "val")

    def test_step(self, batch: BatchedLabeledPathContext, batch_idx: int) -> Dict:  # type: ignore
        return self._shared_step(batch, "test")

    # ========== ON EPOCH END ==========

    def _shared_epoch_end(self, outputs: List[Dict], step: str):
        with torch.no_grad():
            mean_loss = torch.stack([out["loss"] for out in outputs]).mean()
            accuracy = self.__metrics[f"{step}_acc"].compute()
            log = {f"{step}/loss": mean_loss, f"{step}/accuracy": accuracy}
        self.log_dict(log, on_step=False, on_epoch=True)

    def training_epoch_end(self, outputs: List[Dict]):
        self._shared_epoch_end(outputs, "train")

    def validation_epoch_end(self, outputs: List[Dict]):
        self._shared_epoch_end(outputs, "val")

    def test_epoch_end(self, outputs: List[Dict]):
        self._shared_epoch_end(outputs, "test")
