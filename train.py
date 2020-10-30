from argparse import ArgumentParser
from os.path import join

import torch
from pytorch_lightning import Trainer, seed_everything, LightningModule, LightningDataModule
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from configs import Code2SeqConfig, Code2ClassConfig, Code2SeqTestConfig, Code2ClassTestConfig
from configs.parts import ModelHyperParameters
from dataset import PathContextDataModule
from model import Code2Seq, Code2Class
from utils.common import SEED, DATA_FOLDER, VOCABULARY_NAME
from utils.vocabulary import Vocabulary


def train(
    model: LightningModule,
    data_module: LightningDataModule,
    hyper_parameters: ModelHyperParameters,
    wandb_project: str,
    log_offline: bool = False,
    resume_from_checkpoint: str = None,
):
    seed_everything(SEED)

    # define logger
    wandb_logger = WandbLogger(project=wandb_project, log_model=True, offline=log_offline)
    wandb_logger.watch(model)
    # define model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        filepath=join(wandb_logger.experiment.dir, "{epoch:02d}-{val_loss:.4f}"),
        period=hyper_parameters.save_every_epoch,
    )
    # define early stopping callback
    early_stopping_callback = EarlyStopping(
        patience=hyper_parameters.patience, monitor="val/loss", verbose=True, mode="min"
    )
    # use gpu if it exists
    gpu = 1 if torch.cuda.is_available() else None
    # define learning rate logger
    lr_logger = LearningRateMonitor()
    trainer = Trainer(
        max_epochs=hyper_parameters.n_epochs,
        gradient_clip_val=hyper_parameters.clip_norm,
        deterministic=True,
        check_val_every_n_epoch=hyper_parameters.val_every_epoch,
        log_every_n_steps=hyper_parameters.log_every_epoch,
        logger=wandb_logger,
        resume_from_checkpoint=resume_from_checkpoint,
        gpus=gpu,
        callbacks=[lr_logger, early_stopping_callback, checkpoint_callback],
    )

    trainer.fit(model=model, datamodule=data_module)
    trainer.test()


def train_code2seq(
    config: Code2SeqConfig,
    dataset_name: str,
    num_workers: int = 0,
    log_offline: bool = False,
    resume_from_checkpoint: str = None,
):
    vocabulary = Vocabulary.load_vocabulary(join(DATA_FOLDER, dataset_name, VOCABULARY_NAME))
    model = Code2Seq(config, vocabulary)
    data_module = PathContextDataModule(
        dataset_name, vocabulary, config.data_processing, config.hyper_parameters, num_workers
    )
    train(model, data_module, config.hyper_parameters, f"code2seq-{dataset_name}", log_offline, resume_from_checkpoint)


def train_code2class(
    config: Code2ClassConfig,
    dataset_name: str,
    num_workers: int = 0,
    log_offline: bool = False,
    resume_from_checkpoint: str = None,
):
    vocabulary = Vocabulary.load_vocabulary(join(DATA_FOLDER, dataset_name, VOCABULARY_NAME))
    model = Code2Class(config, vocabulary)
    data_module = PathContextDataModule(
        dataset_name, vocabulary, config.data_processing, config.hyper_parameters, num_workers
    )
    train(
        model, data_module, config.hyper_parameters, f"code2class-{dataset_name}", log_offline, resume_from_checkpoint
    )


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("dataset_name", type=str)
    arg_parser.add_argument("model", choices=["code2seq", "code2class"])
    arg_parser.add_argument("--num_workers", type=int, default=0)
    arg_parser.add_argument("--test", action="store_true")
    arg_parser.add_argument("--resume", type=str, default=None)
    args = arg_parser.parse_args()

    if args.model == "code2seq":
        _config = Code2SeqTestConfig() if args.test else Code2SeqConfig()
        train_code2seq(_config, args.dataset_name, args.num_workers, args.test, args.resume)
    elif args.model == "code2class":
        _config = Code2ClassTestConfig() if args.test else Code2ClassConfig()
        train_code2class(_config, args.dataset_name, args.num_workers, args.test, args.resume)
    else:
        print(f'Unknown model: {args.model}, try on of: "code2seq", "code2class"')
