from abc import abstractmethod
from typing import Tuple, Dict, List

from pytorch_lightning.core.lightning import LightningModule
from torch.optim import Adam, Optimizer, SGD
from torch.optim.lr_scheduler import _LRScheduler, LambdaLR

from configs.parts import ModelHyperParameters
from utils.vocabulary import Vocabulary


class BaseCodeModel(LightningModule):
    def __init__(self, hyperparams: ModelHyperParameters, vocab: Vocabulary):
        super().__init__()
        self.hyperparams = hyperparams
        self.vocab = vocab

    @abstractmethod
    def _general_epoch_end(self, outputs: List[Dict], loss_key: str, group: str) -> Dict:
        pass

    # ===== OPTIMIZERS =====

    def configure_optimizers(self) -> Tuple[List[Optimizer], List[_LRScheduler]]:
        optimizer: Optimizer
        if self.hyperparams.optimizer == "Momentum":
            # using the same momentum value as in original realization by Alon
            optimizer = SGD(
                self.parameters(),
                self.hyperparams.learning_rate,
                momentum=0.95,
                nesterov=self.hyperparams.nesterov,
                weight_decay=self.hyperparams.weight_decay,
            )
        elif self.hyperparams.optimizer == "Adam":
            optimizer = Adam(
                self.parameters(), self.hyperparams.learning_rate, weight_decay=self.hyperparams.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer name: {self.hyperparams.optimizer}, try one of: Adam, Momentum")
        scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: self.hyperparams.decay_gamma ** epoch)
        return [optimizer], [scheduler]

    # ===== ON EPOCH END =====

    def training_epoch_end(self, outputs: List[Dict]):
        self._general_epoch_end(outputs, "loss", "train")

    def validation_epoch_end(self, outputs: List[Dict]):
        self._general_epoch_end(outputs, "val_loss", "val")

    def test_epoch_end(self, outputs: List[Dict]):
        self._general_epoch_end(outputs, "test_loss", "test")
