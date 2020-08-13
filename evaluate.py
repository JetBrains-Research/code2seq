from argparse import ArgumentParser
from math import ceil
from multiprocessing import cpu_count

import torch
from pytorch_lightning import Trainer, seed_everything

from dataset import create_dataloader
from model import Code2Seq

SEED = 7


def evaluate(checkpoint: str, data: str = None, batch_size: int = None):
    seed_everything(SEED)
    model = Code2Seq.load_from_checkpoint(checkpoint_path=checkpoint)
    batch_size = batch_size or model.hyperparams.test_batch_size
    data = data or model.hyperparams.test_data_path
    gpu = 1 if torch.cuda.is_available() else None
    data_loader, n_samples = create_dataloader(
        data, model.hyperparams.max_context, False, False, batch_size, cpu_count(),
    )
    print(f"approximate number of steps for test is {ceil(n_samples / batch_size)}")
    trainer = Trainer(gpus=gpu)
    trainer.test(model, test_dataloaders=data_loader)


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("checkpoint", type=str)
    arg_parser.add_argument("--data", type=str, default=None)
    arg_parser.add_argument("--batch-size", type=int, default=None)

    args = arg_parser.parse_args()

    evaluate(args.checkpoint, args.data, args.batch_size)
