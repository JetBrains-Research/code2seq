from argparse import ArgumentParser
from os.path import join
from typing import Union, Tuple

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from configs import Code2SeqConfig, Code2ClassConfig, Code2SeqTestConfig, Code2ClassTestConfig
from dataset import PathContextDataModule
from model import Code2Seq, Code2Class
from utils.common import SEED, DATA_FOLDER, VOCABULARY_NAME
from utils.vocabulary import Vocabulary


def init_code2seq_model(vocabulary: Vocabulary, is_test: bool) -> Tuple[Code2SeqConfig, Code2Seq]:
    config = Code2SeqTestConfig() if is_test else Code2SeqConfig()
    model = Code2Seq(config, vocabulary)
    return config, model


def init_code2class_model(vocabulary: Vocabulary, is_test: bool) -> Tuple[Code2ClassConfig, Code2Class]:
    config = Code2ClassTestConfig() if is_test else Code2ClassConfig()
    model = Code2Class(config, vocabulary)
    return config, model


def train(
    dataset_name: str, model_name: str, num_workers: int = 0, is_test: bool = False, resume_from_checkpoint: str = None
):
    seed_everything(SEED)

    # load vocabulary
    vocabulary = Vocabulary.load_vocabulary(join(DATA_FOLDER, dataset_name, VOCABULARY_NAME))

    # initialize model
    config: Union[Code2SeqConfig, Code2ClassConfig]
    if model_name == "code2seq":
        config, model = init_code2seq_model(vocabulary, is_test)
    elif model_name == "code2class":
        config, model = init_code2class_model(vocabulary, is_test)
    else:
        raise ValueError(f"Model {model_name} is not supported")

    # initialize data module
    data_module = PathContextDataModule(
        dataset_name, vocabulary, config.data_processing, config.hyper_parameters, num_workers
    )

    # define logger
    wandb_logger = WandbLogger(project=f"{model_name}-{dataset_name}", log_model=True, offline=is_test)
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
    lr_logger = LearningRateMonitor()
    trainer = Trainer(
        max_epochs=config.hyper_parameters.n_epochs,
        gradient_clip_val=config.hyper_parameters.clip_norm,
        deterministic=True,
        check_val_every_n_epoch=config.hyper_parameters.val_every_epoch,
        log_every_n_steps=config.hyper_parameters.log_every_epoch,
        logger=wandb_logger,
        resume_from_checkpoint=resume_from_checkpoint,
        gpus=gpu,
        callbacks=[lr_logger, early_stopping_callback, checkpoint_callback],
    )

    trainer.fit(model=model, datamodule=data_module)

    trainer.test()


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("data", type=str)
    arg_parser.add_argument("model", choices=["code2seq", "code2class"])
    arg_parser.add_argument("--n_workers", type=int, default=0)
    arg_parser.add_argument("--test", action="store_true")
    arg_parser.add_argument("--resume", type=str, default=None)
    args = arg_parser.parse_args()

    train(args.data, args.model, args.n_workers, args.test, args.resume)
