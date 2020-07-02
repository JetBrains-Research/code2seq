import pickle
from argparse import ArgumentParser
from os import mkdir
from os.path import join

import wandb
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from configs import get_code2seq_default_config, get_code2seq_test_config
from model import Code2Seq

DATA_FOLDER = "data"
SEED = 7


def train(dataset_name: str, is_test: bool):
    seed_everything(SEED)
    dataset_main_folder = join(DATA_FOLDER, dataset_name)
    with open(join(dataset_main_folder, "vocabulary.pkl"), "rb") as pkl_file:
        vocab = pickle.load(pkl_file)

    config_function = get_code2seq_test_config if is_test else get_code2seq_default_config
    config = config_function(dataset_main_folder)

    model = Code2Seq(config, vocab)

    # define logger
    wandb_logger = WandbLogger(project=f"code2seq-{dataset_name}", offline=is_test)
    wandb_logger.watch(model, log="all", log_freq=config.log_every_epoch)
    # define model checkpoint callback
    checkpoint_path = join(wandb.run.dir, "checkpoints")
    mkdir(checkpoint_path)
    model_checkpoint_callback = ModelCheckpoint(
        filepath=join(checkpoint_path, "{epoch:02d}-{val_loss:.4f}"),
        verbose=True,
        period=config.save_every_epoch,
        save_top_k=-1,
    )
    # define early stopping callback
    early_stopping_callback = EarlyStopping(patience=config.patience, verbose=True)
    trainer = Trainer(
        deterministic=True,
        check_val_every_n_epoch=config.val_every_epoch,
        logger=wandb_logger,
        checkpoint_callback=model_checkpoint_callback,
        early_stop_callback=early_stopping_callback,
    )
    trainer.fit(model)


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("data", type=str)
    arg_parser.add_argument("--test", action="store_true")
    args = arg_parser.parse_args()

    train(args.data, args.test)
