from argparse import ArgumentParser
from math import ceil
from multiprocessing import cpu_count

from pytorch_lightning import Trainer

from dataset import create_dataloader
from model import Code2Seq


def evaluate(checkpoint: str, data: str = None):
    model = Code2Seq.load_from_checkpoint(checkpoint_path=checkpoint)
    trainer = Trainer()
    if data is not None:
        data_loader, n_samples = create_dataloader(
            data, model.config.max_context, False, False, model.config.test_batch_size, cpu_count()
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
