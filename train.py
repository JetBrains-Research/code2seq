from argparse import ArgumentParser
from os.path import join
from typing import Tuple

import torch
from omegaconf import DictConfig
from pytorch_lightning import seed_everything, Trainer, LightningModule, LightningDataModule
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from dataset import PathContextDataModule, TypedPathContextDataModule
from model import Code2Seq, Code2Class, TypedCode2Seq
from utils.callback import UploadCheckpointCallback, PrintEpochResultCallback
from utils.common import print_config, filter_warnings, get_config
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


def train(config: DictConfig, resume_from_checkpoint: str = None):
    filter_warnings()
    print_config(config)
    seed_everything(config.seed)

    known_models = {"code2seq": get_code2seq, "code2class": get_code2class, "typed-code2seq": get_typed_code2seq}
    if config.name not in known_models:
        print(f"Unknown model: {config.name}, try on of {known_models.keys()}")

    vocabulary = Vocabulary.load_vocabulary(join(config.data_folder, config.dataset.name, config.vocabulary_name))
    model, data_module = known_models[config.name](config, vocabulary)

    # define logger
    wandb_logger = WandbLogger(
        project=f"{config.name}-{config.dataset.name}", log_model=True, offline=config.log_offline
    )
    wandb_logger.watch(model)
    # define model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=wandb_logger.experiment.dir,
        filename="{epoch:02d}-{val_loss:.4f}",
        period=config.save_every_epoch,
        save_top_k=-1,
    )
    upload_checkpoint_callback = UploadCheckpointCallback(wandb_logger.experiment.dir)
    # define early stopping callback
    early_stopping_callback = EarlyStopping(
        patience=config.hyper_parameters.patience, monitor="val_loss", verbose=True, mode="min"
    )
    # define callback for printing intermediate result
    print_epoch_result_callback = PrintEpochResultCallback("train", "val")
    # use gpu if it exists
    gpu = 1 if torch.cuda.is_available() else None
    # define learning rate logger
    lr_logger = LearningRateMonitor("step")
    trainer = Trainer(
        max_epochs=config.hyper_parameters.n_epochs,
        gradient_clip_val=config.hyper_parameters.clip_norm,
        deterministic=True,
        check_val_every_n_epoch=config.val_every_epoch,
        log_every_n_steps=config.log_every_epoch,
        logger=wandb_logger,
        gpus=gpu,
        progress_bar_refresh_rate=config.progress_bar_refresh_rate,
        callbacks=[
            lr_logger,
            early_stopping_callback,
            checkpoint_callback,
            upload_checkpoint_callback,
            print_epoch_result_callback,
        ],
        resume_from_checkpoint=resume_from_checkpoint,
    )

    trainer.fit(model=model, datamodule=data_module)
    trainer.test()


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("model", type=str)
    arg_parser.add_argument("--dataset", type=str, default=None)
    arg_parser.add_argument("--offline", action="store_true")
    arg_parser.add_argument("--resume", type=str, default=None)
    args = arg_parser.parse_args()

    _config = get_config(args.model, args.dataset, log_offline=args.offline)
    train(_config, args.resume)
