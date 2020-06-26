from typing import Tuple, Dict, List

import torch
from pytorch_lightning.core.lightning import LightningModule
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader

from configs import Code2SeqConfig
from dataset.path_context_dataset import PathContextDataset, collate_path_contexts


class Code2Seq(LightningModule):
    def __init__(self, config: Code2SeqConfig):
        super().__init__()
        self.config = config

    def forward(self, samples: Dict[str, torch.Tensor]) -> torch.Tensor:
        pass

    def configure_optimizers(self) -> Optimizer:
        return Adam(self.parameters(), self.config.learning_rate)

    # ===== TRAIN BLOCK =====

    def train_dataloader(self) -> DataLoader:
        dataset = PathContextDataset(self.config.train_data_path, self.config.shuffle_data)
        data_loader = DataLoader(dataset, batch_size=self.config.batch_size, collate_fn=collate_path_contexts)
        return data_loader

    def training_step(self, batch: Tuple[Dict[str, torch.Tensor], torch.Tensor], batch_idx: int) -> Dict:
        pass

    # ===== VALIDATION BLOCK =====

    def val_dataloader(self) -> DataLoader:
        dataset = PathContextDataset(self.config.val_data_path, False)
        data_loader = DataLoader(dataset, batch_size=self.config_val_batch_size, collate_fn=collate_path_contexts)
        return data_loader

    def validation_step(self, batch: Tuple[Dict[str, torch.Tensor], torch.Tensor], batch_idx: int) -> Dict:
        pass

    def validation_epoch_end(self, outputs: List[Dict]) -> Dict:
        pass
