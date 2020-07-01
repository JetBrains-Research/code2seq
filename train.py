import pickle
from argparse import ArgumentParser
from os.path import join

from pytorch_lightning import Trainer

from configs import get_code2seq_default_config, get_code2seq_test_config
from model import Code2Seq

DATA_FOLDER = "data"


def train(dataset_name: str, is_test: bool):
    dataset_main_folder = join(DATA_FOLDER, dataset_name)
    with open(join(dataset_main_folder, "vocabulary.pkl"), "rb") as pkl_file:
        vocab = pickle.load(pkl_file)

    config_function = get_code2seq_test_config if is_test else get_code2seq_default_config
    config = config_function(dataset_main_folder)

    model = Code2Seq(config, vocab)
    trainer = Trainer()
    trainer.fit(model)


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("data", type=str)
    arg_parser.add_argument("--test", action="store_true")
    args = arg_parser.parse_args()

    train(args.data, args.test)
