from typing import Tuple, Dict

from pytorch_lightning.core.lightning import LightningModule
from torch.utils.data import DataLoader

from configs import Code2SeqConfig
from dataset.path_context_dataset import PathContextDataset, collate_path_contexts


class Code2Seq(LightningModule):
    def __init__(self, config: Code2SeqConfig):
        super().__init__()
        self.config = config

    def forward(self):
        pass

    def training_step(self, batch: Tuple, batch_idx: int) -> Dict:
        pass

    def configure_optimizers(self):
        pass

    def train_dataloader(self) -> DataLoader:
        dataset = PathContextDataset(self.config.train_data_path, self.config.shuffle_data)
        data_loader = DataLoader(dataset, batch_size=self.config.batch_size, collate_fn=collate_path_contexts)
        return data_loader

    def val_dataloader(self) -> DataLoader:
        dataset = PathContextDataset(self.config.val_data_path, False)
        data_loader = DataLoader(dataset, batch_size=self.config_val_batch_size, collate_fn=collate_path_contexts)
        return data_loader
