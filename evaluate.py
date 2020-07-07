from argparse import ArgumentParser
from math import ceil
from multiprocessing import cpu_count
from os.path import join

import torch
from pytorch_lightning import Trainer, seed_everything

from dataset import create_dataloader
from model import Code2Seq


DATA_FOLDER = "data"
SEED = 7


def evaluate(checkpoint: str, data: str = None):
    seed_everything(SEED)
    model = Code2Seq.load_from_checkpoint(checkpoint_path=checkpoint)
    gpu = 1 if torch.cuda.is_available() else None
    trainer = Trainer(gpus=gpu)
    if data is not None:
        data_loader, n_samples = create_dataloader(
            join(DATA_FOLDER, data), model.config.max_context, False, False, model.config.test_batch_size, cpu_count()
        )
        print(f"approximate number of steps for test is {ceil(n_samples / model.config.test_batch_size)}")
        trainer.test(model, test_dataloaders=data_loader)
    else:
        trainer.test(model)


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("checkpoint", type=str)
    arg_parser.add_argument("--data", type=str, default=None)
    args = arg_parser.parse_args()

    evaluate(args.checkpoint, args.data)
