from argparse import ArgumentParser
from multiprocessing import cpu_count

import torch
from pytorch_lightning import Trainer, seed_everything

from dataset import PathContextDataModule
from model import Code2Seq
from utils.common import SEED


def evaluate(checkpoint: str, data: str, batch_size: int = None):
    seed_everything(SEED)
    model = Code2Seq.load_from_checkpoint(checkpoint_path=checkpoint)
    config = model.get_config()
    if batch_size is not None:
        config.hyper_parameters.test_batch_size = batch_size
    vocabulary = model.get_vocabulary()

    data_module = PathContextDataModule(data, vocabulary, config.data_processing, config.hyper_parameters, cpu_count())

    gpu = 1 if torch.cuda.is_available() else None
    trainer = Trainer(gpus=gpu)
    trainer.test(model, datamodule=data_module)


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("checkpoint", type=str)
    arg_parser.add_argument("data", type=str)
    arg_parser.add_argument("--batch-size", type=int, default=None)

    args = arg_parser.parse_args()

    evaluate(args.checkpoint, args.data, args.batch_size)
