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
    arg_parser.add_argument("mode", help="Mode to run script", choices=["train", "test", "predict"])
    arg_parser.add_argument("-c", "--config", help="Path to YAML configuration file", type=str)
    arg_parser.add_argument(
        "-p", "--pretrained", help="Path to pretrained model", type=str, required=False, default=None
    )
    arg_parser.add_argument(
        "-o", "--output", help="Output file for predictions", type=str, required=False, default=None
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


def save_predictions(model_path: str, config: DictConfig, output_path: str):
    filter_warnings()

    data_module = CommentPathContextDataModule(config.data_folder, config.data)
    tokenizer = data_module.vocabulary.tokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    code2seq = CommentCode2Seq.load_from_checkpoint(model_path)
    code2seq.to(device)
    code2seq.eval()

    with open(output_path, "w") as f:
        for batch in data_module.test_dataloader():
            data_module.transfer_batch_to_device(batch, device, 0)
            logits, _ = code2seq.logits_from_batch(batch, None)

            predictions = logits[:-1].argmax(-1)
            targets = batch.labels[1:]

            batch_size = targets.shape[1]
            for batch_idx in range(batch_size):
                target_seq = [token.item() for token in targets[:, batch_idx]]
                predicted_seq = [token.item() for token in predictions[:, batch_idx]]

                target_str = tokenizer.decode(target_seq, skip_special_tokens=True)
                predicted_str = tokenizer.decode(predicted_seq, skip_special_tokens=True)

                if target_str == "":
                    continue

                print(target_str.replace(" ", "|"), predicted_str.replace(" ", "|"), file=f)


if __name__ == "__main__":
    __arg_parser = configure_arg_parser()
    __args = __arg_parser.parse_args()

    __config = cast(DictConfig, OmegaConf.load(__args.config))
    seed_everything(__config.seed)
    if __args.mode == "train":
        train_code2seq(__config)
    else:
        assert __args.pretrained is not None
        if __args.mode == "test":
            test_code2seq(__args.pretrained, __config)
        else:
            save_predictions(__args.pretrained, __config, __args.output)
