from os.path import join
from typing import Tuple

import torch
from hydra import main
from omegaconf import DictConfig
from pytorch_lightning import seed_everything, Trainer, LightningModule, LightningDataModule
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from dataset import PathContextDataModule, TypedPathContextDataModule
from model import Code2Seq, Code2Class, TypedCode2Seq
from utils.vocabulary import Vocabulary


def get_code2seq(config: DictConfig, vocabulary: Vocabulary) -> Tuple[LightningModule, LightningDataModule]:
    model = Code2Seq(config, vocabulary)
    data_module = PathContextDataModule(config, vocabulary)
    return model, data_module


def get_code2class(config: DictConfig, vocabulary: Vocabulary) -> Tuple[LightningModule, LightningDataModule]:
    model = Code2Class(config, vocabulary)
    data_module = PathContextDataModule(config, vocabulary)
    return model, data_module


def get_typed_code2seq(config: DictConfig, vocabulary: Vocabulary) -> Tuple[LightningModule, LightningDataModule]:
    model = TypedCode2Seq(config, vocabulary)
    data_module = TypedPathContextDataModule(config, vocabulary)
    return model, data_module


@main(config_path="configs", config_name="train")
def train(config: DictConfig):
    known_models = {"code2seq": get_code2seq, "code2class": get_code2class, "typed_code2seq": get_typed_code2seq}
    if config.name not in known_models:
        print(f"Unknown model: {config.name}, try on of {known_models.keys()}")

    vocabulary = Vocabulary.load_vocabulary(join(config.data_folder, config.dataset.name, config.vocabulary_name))
    model, data_module = known_models[config.name](config, vocabulary)

    seed_everything(config.seed)

    # define logger
    wandb_logger = WandbLogger(
        project=f"{config.name}-{config.dataset.name}", log_model=True, offline=config.log_offline
    )
    wandb_logger.watch(model)
    # define model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        filepath=join(wandb_logger.experiment.dir, "{epoch:02d}-{val_loss:.4f}"),
        period=config.hyper_parameters.save_every_epoch,
    )
    # define early stopping callback
    early_stopping_callback = EarlyStopping(
        patience=config.hyper_parameters.patience, monitor="val/loss", verbose=True, mode="min"
    )
    # use gpu if it exists
    gpu = 1 if torch.cuda.is_available() else None
    # define learning rate logger
    lr_logger = LearningRateMonitor(config.hyper_parameters.log_every_epoch)
    trainer = Trainer(
        max_epochs=config.hyper_parameters.n_epochs,
        gradient_clip_val=config.hyper_parameters.clip_norm,
        deterministic=True,
        check_val_every_n_epoch=config.hyper_parameters.val_every_epoch,
        log_every_n_steps=config.hyper_parameters.log_every_epoch,
        logger=wandb_logger,
        gpus=gpu,
        callbacks=[lr_logger, early_stopping_callback, checkpoint_callback],
    )

    trainer.fit(model=model, datamodule=data_module)
    trainer.test()


if __name__ == "__main__":
    train()
