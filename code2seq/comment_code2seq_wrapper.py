from argparse import ArgumentParser
from typing import cast

import torch
from commode_utils.common import print_config
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything

from code2seq.data.comment_path_context_data_module import CommentPathContextDataModule
from code2seq.model.comment_code2seq import CommentCode2Seq
from code2seq.utils.common import filter_warnings
from code2seq.utils.test import test
from code2seq.utils.train import train


def configure_arg_parser() -> ArgumentParser:
    arg_parser = ArgumentParser()
    arg_parser.add_argument("mode", help="Mode to run script", choices=["train", "test"])
    arg_parser.add_argument("-c", "--config", help="Path to YAML configuration file", type=str)
    arg_parser.add_argument(
        "-p", "--pretrained", help="Path to pretrained model", type=str, required=False, default=None
    )
    return arg_parser


def train_code2seq(config: DictConfig):
    filter_warnings()

    if config.print_config:
        print_config(config, fields=["model", "data", "train", "optimizer"])

    # Load data module
    data_module = CommentPathContextDataModule(config.data_folder, config.data)

    # Load model
    code2seq = CommentCode2Seq(config.model, config.optimizer, data_module.vocabulary, config.train.teacher_forcing)

    train(code2seq, data_module, config)


def test_code2seq(model_path: str, config: DictConfig):
    filter_warnings()

    # Load data module
    data_module = CommentPathContextDataModule(config.data_folder, config.data)

    # Load model
    code2seq = CommentCode2Seq.load_from_checkpoint(model_path, map_location=torch.device("cpu"))

    test(code2seq, data_module, config.seed)


if __name__ == "__main__":
    __arg_parser = configure_arg_parser()
    __args = __arg_parser.parse_args()

    __config = cast(DictConfig, OmegaConf.load(__args.config))
    seed_everything(__config.seed)
    if __args.mode == "train":
        train_code2seq(__config)
    else:
        assert __args.pretrained is not None
        test_code2seq(__args.pretrained, __config)
