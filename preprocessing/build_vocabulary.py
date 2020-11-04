import pickle
from argparse import ArgumentParser
from collections import Counter
from os import path
from os.path import join, exists
from typing import List, Dict

from tqdm import tqdm

from configs import Code2SeqConfig, Code2ClassConfig
from configs.parts import DataProcessingConfig
from utils.common import DATA_FOLDER, VOCABULARY_NAME, SOS, EOS, PAD, UNK, TRAIN_HOLDOUT
from utils.vocabulary import Vocabulary
from utils.converting import parse_token
from utils.filesystem import count_lines_in_file


_config_switcher = {"code2class": Code2ClassConfig, "code2seq": Code2SeqConfig}


def _counter_to_dict(values: Counter, n_most_common: int = None, additional_values: List[str] = None) -> Dict[str, int]:
    dict_values = []
    if additional_values is not None:
        dict_values += additional_values
    dict_values += list(zip(*values.most_common(n_most_common)))[0]
    return {value: i for i, value in enumerate(dict_values)}


def _counters_to_vocab(
    config: DataProcessingConfig, token_counter: Counter, target_counter: Counter, type_counter: Counter
) -> Vocabulary:
    names_additional_tokens = [SOS, EOS, PAD, UNK] if config.wrap_name else [PAD, UNK]
    token_to_id = _counter_to_dict(token_counter, config.subtoken_vocab_max_size, names_additional_tokens)
    target_additional_tokens = [SOS, EOS, PAD, UNK] if config.wrap_target else [PAD, UNK]
    label_to_id = _counter_to_dict(target_counter, config.target_vocab_max_size, target_additional_tokens)
    paths_additional_tokens = [SOS, EOS, PAD, UNK] if config.wrap_path else [PAD, UNK]
    type_to_id = _counter_to_dict(type_counter, None, paths_additional_tokens)
    return Vocabulary(token_to_id=token_to_id, label_to_id=label_to_id, type_to_id=type_to_id)


def collect_vocabulary(config: DataProcessingConfig, dataset_name: str) -> Vocabulary:
    target_counter = Counter()
    token_counter = Counter()
    type_counter = Counter()
    train_data_path = path.join(DATA_FOLDER, dataset_name, f"{dataset_name}.{TRAIN_HOLDOUT}.c2s")
    with open(train_data_path, "r") as train_file:
        for line in tqdm(train_file, total=count_lines_in_file(train_data_path)):
            label, *path_contexts = line.split()
            target_counter.update(parse_token(label, config.split_target))
            cur_tokens = []
            cur_types = []
            for path_context in path_contexts:
                from_token, path_types, to_token = path_context.split(",")
                cur_tokens += parse_token(from_token, config.split_names) + parse_token(to_token, config.split_names)
                cur_types += path_types.split("|")
            token_counter.update(cur_tokens)
            type_counter.update(cur_types)
    return _counters_to_vocab(config, token_counter, target_counter, type_counter)


def convert_vocabulary(config: DataProcessingConfig, original_vocabulary_path: str) -> Vocabulary:
    with open(original_vocabulary_path, "rb") as dict_file:
        subtoken_to_count = Counter(pickle.load(dict_file))
        node_to_count = Counter(pickle.load(dict_file))
        target_to_count = Counter(pickle.load(dict_file))
    return _counters_to_vocab(config, subtoken_to_count, target_to_count, node_to_count)


def preprocess(problem: str, dataset_name: str, convert_path: str = None):
    if convert_path is not None and not exists(convert_path):
        raise ValueError(f"There is no file for converting: {convert_path}")
    vocabulary_path = join(DATA_FOLDER, dataset_name, VOCABULARY_NAME)

    if problem not in _config_switcher:
        raise ValueError(f"Unknown problem {problem}, specify one of: {_config_switcher.keys()}")
    config = _config_switcher[problem].data_processing

    if convert_path is None:
        vocabulary = collect_vocabulary(config, dataset_name)
    else:
        vocabulary = convert_vocabulary(config, convert_path)
    vocabulary.dump_vocabulary(vocabulary_path)


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("problem", type=str)
    arg_parser.add_argument("dataset_name", type=str)
    arg_parser.add_argument("--convert", type=str, default=None)
    args = arg_parser.parse_args()

    preprocess(args.problem, args.dataset_name, args.convert)
