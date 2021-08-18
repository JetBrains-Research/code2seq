from typing import Optional

import torch
from pytorch_lightning import Trainer, seed_everything, LightningModule, LightningDataModule


def test(model: LightningModule, data_module: LightningDataModule, seed: Optional[int] = None):
    seed_everything(seed)
    gpu = 1 if torch.cuda.is_available() else None
    trainer = Trainer(gpus=gpu)
    trainer.test(model, datamodule=data_module)
