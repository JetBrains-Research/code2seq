from argparse import ArgumentParser
from multiprocessing import cpu_count

import torch
from pytorch_lightning import Trainer, seed_everything

from dataset import PathContextDataModule
from model import Code2Seq


def evaluate(checkpoint: str, data: str, batch_size: int = None):
    model = Code2Seq.load_from_checkpoint(checkpoint_path=checkpoint)
    config = model.config
    vocabulary = model.vocabulary
    if batch_size is not None:
        config.hyper_parameters.test_batch_size = batch_size

    data_module = PathContextDataModule(config, vocabulary)

    seed_everything(config.seed)
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
