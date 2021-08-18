from argparse import ArgumentParser
from typing import cast

import torch
from commode_utils.common import print_config
from omegaconf import DictConfig, OmegaConf

from src.data import TypedPathContextDataModule
from src.model import TypedCode2Seq
from src.utils.common import filter_warnings
from src.utils.test import test
from src.utils.train import train


def configure_arg_parser() -> ArgumentParser:
    arg_parser = ArgumentParser()
    arg_parser.add_argument("mode", help="Mode to run script", choices=["train", "test"])
    arg_parser.add_argument("-c", "--config", help="Path to YAML configuration file", type=str)
    return arg_parser


def train_typed_code2seq(config: DictConfig):
    filter_warnings()

    if config.print_config:
        print_config(config, fields=["model", "data", "train", "optimizer"])

    # Load data module
    data_module = TypedPathContextDataModule(config.data, config.data_folder)
    data_module.prepare_data()
    data_module.setup()

    # Load model
    typed_code2seq = TypedCode2Seq(config.model, config.optimizer, data_module.vocabulary, config.train.teacher_forcing)

    train(typed_code2seq, data_module, config)


def test_typed_code2seq(config: DictConfig):
    filter_warnings()

    # Load data module
    data_module = TypedPathContextDataModule(config.data, config.data_folder)
    data_module.prepare_data()
    data_module.setup()

    # Load model
    typed_code2seq = TypedCode2Seq.load_from_checkpoint(config.checkpoint, map_location=torch.device("cpu"))

    test(typed_code2seq, data_module, config.seed)


if __name__ == "__main__":
    __arg_parser = configure_arg_parser()
    __args = __arg_parser.parse_args()

    __config = cast(DictConfig, OmegaConf.load(__args.config))
    if __args.mode == "train":
        train_typed_code2seq(__config)
    else:
        test_typed_code2seq(__config)
