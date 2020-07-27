from argparse import ArgumentParser
from dataclasses import asdict
from os.path import join

import torch
import wandb
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateLogger
from pytorch_lightning.loggers import WandbLogger

from configs import (
    get_code2seq_default_config,
    get_code2seq_test_config,
    get_code2class_test_config,
    get_code2class_default_config,
)
from dataset import Vocabulary
from model import Code2Seq, Code2Class

DATA_FOLDER = "data"
SEED = 7


def train(
    dataset_name: str, model_name: str, num_workers: int = 0, is_test: bool = False, resume_from_checkpoint: str = None
):
    seed_everything(SEED)
    dataset_main_folder = join(DATA_FOLDER, dataset_name)
    vocab = Vocabulary.load(join(dataset_main_folder, "vocabulary.pkl"))

    if model_name == "code2seq":
        config_function = get_code2seq_test_config if is_test else get_code2seq_default_config
        hyperparams, encoder_config, decoder_config = config_function(dataset_main_folder)
        model = Code2Seq(hyperparams, vocab, num_workers, encoder_config=encoder_config, decoder_config=decoder_config)
    elif model_name == "code2class":
        config_function = get_code2class_test_config if is_test else get_code2class_default_config
        hyperparams, encoder_config, decoder_config = config_function(dataset_main_folder)
        model = Code2Class(
            hyperparams, vocab, num_workers, encoder_config=encoder_config, decoder_config=decoder_config
        )
    else:
        raise ValueError(f"Model {model_name} is not supported")

    # define logger
    wandb_logger = WandbLogger(project=f"code2seq-{dataset_name}", log_model=True, offline=is_test)
    wandb_logger.watch(model)
    wandb_logger.log_hyperparams(asdict(hyperparams))
    wandb_logger.log_hyperparams(asdict(encoder_config))
    wandb_logger.log_hyperparams(asdict(decoder_config))
    # define model checkpoint callback
    model_checkpoint_callback = ModelCheckpoint(
        filepath=join(wandb.run.dir, "{epoch:02d}-{val_loss:.4f}"), period=config.save_every_epoch, save_top_k=3,
    )
    # define early stopping callback
    early_stopping_callback = EarlyStopping(patience=config.patience, verbose=True, mode="min")
    # use gpu if it exists
    gpu = 1 if torch.cuda.is_available() else None
    # define learning rate logger
    lr_logger = LearningRateLogger()
    trainer = Trainer(
        max_epochs=config.n_epochs,
        gradient_clip_val=config.clip_norm,
        deterministic=True,
        check_val_every_n_epoch=config.val_every_epoch,
        row_log_interval=config.log_every_epoch,
        logger=wandb_logger,
        checkpoint_callback=model_checkpoint_callback,
        early_stop_callback=early_stopping_callback,
        resume_from_checkpoint=resume_from_checkpoint,
        gpus=gpu,
        callbacks=[lr_logger],
        reload_dataloaders_every_epoch=True,
    )

    trainer.fit(model)

    trainer.test()


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("data", type=str)
    arg_parser.add_argument("--n_workers", type=int, default=0)
    arg_parser.add_argument("--test", action="store_true")
    arg_parser.add_argument("--resume", type=str, default=None)
    arg_parser.add_argument("--model", type=str, default="code2seq")
    args = arg_parser.parse_args()

    train(args.data, args.model, args.n_workers, args.test, args.resume)
