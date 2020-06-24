import pickle
from argparse import ArgumentParser
from collections import Counter
from os import path

from tqdm import tqdm

from configs import get_preprocessing_config_code2seq_params, PreprocessingConfig
from data_loaders import Vocabulary
from utils.common import SOS, EOS, PAD, UNK


def collect_vocabulary(config: PreprocessingConfig) -> Vocabulary:
    label_counter = Counter()
    token_counter = Counter()
    type_counter = Counter()
    train_data_path = path.join(config.data_path, f"{config.dataset_name}.train.c2s")
    with open(train_data_path, "r") as train_file:
        for line in tqdm(train_file):
            method_name, *path_contexts = line.split()
            label_counter.update(method_name.split("|"))
            cur_tokens = []
            cur_types = []
            for path_context in path_contexts:
                from_token, path_types, to_token = path_context.split(",")
                cur_tokens += from_token.split("|") + to_token.split("|")
                cur_types += path_types.split("|")
            token_counter.update(cur_tokens)
            type_counter.update(cur_types)
    vocab = Vocabulary()
    vocab.add_from_counter(
        "token_to_id",
        token_counter,
        config.subtoken_vocab_max_size,
        [SOS, EOS, PAD, UNK],
    )
    vocab.add_from_counter(
        "label_to_id", label_counter, config.target_vocab_max_size, [SOS, EOS, PAD, UNK]
    )
    vocab.add_from_counter("type_to_id", type_counter, -1, [PAD, UNK])
    return vocab


def preprocess(config: PreprocessingConfig):
    # Collect vocabulary from train holdout if needed
    vocab_path = path.join(config.data_path, "vocabulary.pkl")
    if path.exists(vocab_path):
        with open(vocab_path, "rb") as vocab_file:
            vocab = pickle.load(vocab_file)
    else:
        vocab = collect_vocabulary(config)
        with open(vocab_path, "wb") as vocab_file:
            pickle.dump(vocab, vocab_file)


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("data", type=str)
    args = arg_parser.parse_args()

    preprocess(get_preprocessing_config_code2seq_params(args.data))
