from typing import Dict, List

import torch
import torch.nn.functional as F

from configs import Code2ClassConfig
from dataset import Vocabulary, PathContextBatch
from model.modules import PathEncoder, PathClassifier
from utils.common import PAD
from .base_code_model import BaseCodeModel


class Code2Class(BaseCodeModel):
    def __init__(
        self, config: Code2ClassConfig, vocab: Vocabulary, num_workers: int,
    ):
        super().__init__(config.hyperparams, vocab, num_workers)
        self.encoder = PathEncoder(
            config.encoder_config,
            config.classifier_config.classifier_input_size,
            len(self.vocab.token_to_id),
            self.vocab.token_to_id[PAD],
            len(self.vocab.type_to_id),
            self.vocab.type_to_id[PAD],
        )
        self.classifier = PathClassifier(config.classifier_config, len(self.vocab.label_to_id))

    def forward(self, samples: Dict[str, torch.Tensor], paths_for_label: List[int]) -> torch.Tensor:
        return self.classifier(self.encoder(samples), paths_for_label)

    def _general_epoch_end(self, outputs: List[Dict], loss_key: str, group: str) -> Dict:
        logs = {f"{group}/loss": torch.stack([out[loss_key] for out in outputs]).mean()}
        progress_bar = {k: v for k, v in logs.items() if k in [f"{group}/loss"]}
        return {"val_loss": logs[f"{group}/loss"], "log": logs, "progress_bar": progress_bar}

    def training_step(self, batch: PathContextBatch, batch_idx: int) -> Dict:
        # [batch size; num_classes]
        logits = self(batch.context, batch.contexts_per_label)
        loss = F.cross_entropy(logits, batch.labels.squeeze(0))
        log = {"train/loss": loss}
        return {"loss": loss, "log": log}

    def validation_step(self, batch: PathContextBatch, batch_idx: int) -> Dict:
        # [batch size; num_classes]
        logits = self(batch.context, batch.contexts_per_label)
        loss = F.cross_entropy(logits, batch.labels.squeeze(0))
        return {"val_loss": loss}

    def test_step(self, batch: PathContextBatch, batch_idx: int) -> Dict:
        result = self.validation_step(batch, batch_idx)
        result["test_loss"] = result["val_loss"]
        del result["val_loss"]
        return result
