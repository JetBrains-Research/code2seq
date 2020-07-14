import pickle
from argparse import ArgumentParser
from collections import Counter
from multiprocessing import cpu_count
from os import path
from typing import Tuple, List, Any

from tqdm import tqdm

from configs import get_preprocessing_config_code2seq_params, PreprocessingConfig
from dataset import Vocabulary
from utils.common import UNK, count_lines_in_file, vocab_from_counters

from data_preprocessing.buffer_utils import convert_holdout, DATA_FOLDER


def collect_vocabulary(config: PreprocessingConfig) -> Vocabulary:
    target_counter = Counter()
    token_counter = Counter()
    type_counter = Counter()
    train_data_path = path.join(DATA_FOLDER, config.dataset_name, f"{config.dataset_name}.train.c2s")
    with open(train_data_path, "r") as train_file:
        for line in tqdm(train_file, total=count_lines_in_file(train_data_path)):
            label, *path_contexts = line.split()
            target_counter.update(label.split("|"))
            cur_tokens = []
            cur_types = []
            for path_context in path_contexts:
                from_token, path_types, to_token = path_context.split(",")
                cur_tokens += from_token.split("|") + to_token.split("|")
                cur_types += path_types.split("|")
            token_counter.update(cur_tokens)
            type_counter.update(cur_types)
    return vocab_from_counters(config, token_counter, target_counter, type_counter)


def convert_vocabulary(config: PreprocessingConfig) -> Vocabulary:
    with open(path.join(DATA_FOLDER, config.dataset_name, f"{config.dataset_name}.dict.c2s"), "rb",) as dict_file:
        subtoken_to_count = Counter(pickle.load(dict_file))
        node_to_count = Counter(pickle.load(dict_file))
        target_to_count = Counter(pickle.load(dict_file))
    return vocab_from_counters(config, subtoken_to_count, target_to_count, node_to_count)


def _convert_path_context_to_ids(path_context: str, vocab: Vocabulary) -> Tuple[List[int], List[int], List[int]]:
    from_token, path_types, to_token = path_context.split(",")
    token_unk = vocab.token_to_id[UNK]
    type_unk = vocab.type_to_id[UNK]
    return (
        [vocab.token_to_id.get(_t, token_unk) for _t in from_token.split("|")],
        [vocab.type_to_id.get(_t, type_unk) for _t in path.split("|")],
        [vocab.token_to_id.get(_t, token_unk) for _t in to_token.split("|")],
    )


def _split_context(
    line: str, vocab: Vocabulary, **kwargs: Any
) -> Tuple[List[int], List[Tuple[List[int], List[int], List[int]]]]:
    label, *path_contexts = line.split()
    converted_context = [_convert_path_context_to_ids(pc, vocab) for pc in path_contexts]
    return (
        [vocab.label_to_id.get(_l, vocab.label_to_id[UNK]) for _l in label.split("|")],
        converted_context,
    )


def preprocess(config: PreprocessingConfig, is_vocab_collected: bool, n_jobs: int):
    # Collect vocabulary from train holdout if needed
    vocab_path = path.join(DATA_FOLDER, config.dataset_name, "vocabulary.pkl")
    if path.exists(vocab_path):
        vocab = Vocabulary.load(vocab_path)
    else:
        vocab = collect_vocabulary(config) if is_vocab_collected else convert_vocabulary(config)
        vocab.dump(vocab_path)

    for holdout_name in "train", "test", "val":
        holdout_data_path = path.join(DATA_FOLDER, config.dataset_name, f"{config.dataset_name}.{holdout_name}.c2s",)
        holdout_output_folder = path.join(DATA_FOLDER, config.dataset_name, holdout_name)
        convert_holdout(
            holdout_data_path, holdout_output_folder, vocab, config, n_jobs, _split_context,
        )


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("data", type=str)
    arg_parser.add_argument("--collect-vocabulary", action="store_true")
    arg_parser.add_argument("--n-jobs", type=int, default=None)
    args = arg_parser.parse_args()

    preprocess(
        get_preprocessing_config_code2seq_params(args.data), args.collect_vocabulary, args.n_jobs or cpu_count(),
    )
