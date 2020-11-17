import pickle
from collections import Counter
from os import path
from os.path import join, exists
from sys import argv
from typing import Counter as TypeCounter
from typing import List, Dict

from hydra.experimental import initialize_config_dir, compose
from omegaconf import DictConfig
from tqdm import tqdm

from utils.converting import parse_token
from utils.filesystem import count_lines_in_file, get_config_directory
from utils.vocabulary import Vocabulary, SOS, EOS, PAD, UNK


def _counter_to_dict(values: Counter, n_most_common: int = None, additional_values: List[str] = None) -> Dict[str, int]:
    dict_values = []
    if additional_values is not None:
        dict_values += additional_values
    dict_values += list(zip(*values.most_common(n_most_common)))[0]
    return {value: i for i, value in enumerate(dict_values)}


def _counters_to_vocab(
    dataset_config: DictConfig,
    token_counter: Counter,
    target_counter: Counter,
    node_counter: Counter,
    type_counter: Counter,
) -> Vocabulary:
    additional_tokens = [SOS, EOS, PAD, UNK] if dataset_config.token.is_wrapped else [PAD, UNK]
    token_to_id = _counter_to_dict(token_counter, dataset_config.token.vocabulary_size, additional_tokens)
    additional_targets = [SOS, EOS, PAD, UNK] if dataset_config.target.is_wrapped else [PAD, UNK]
    label_to_id = _counter_to_dict(target_counter, dataset_config.target.vocabulary_size, additional_targets)
    additional_nodes = [SOS, EOS, PAD, UNK] if dataset_config.node.is_wrapped else [PAD, UNK]
    node_to_id = _counter_to_dict(node_counter, dataset_config.node.vocabulary_size, additional_nodes)

    vocabulary = Vocabulary(token_to_id, node_to_id, label_to_id)
    if len(type_counter) > 0:
        additional_types = [SOS, EOS, PAD, UNK] if dataset_config.type.is_wrapped else [PAD, UNK]
        vocabulary.type_to_id = _counter_to_dict(type_counter, dataset_config.type.vocabulary_size, additional_types)
    return vocabulary


def collect_vocabulary(config: DictConfig, dataset_directory: str) -> Vocabulary:
    target_counter: TypeCounter[str] = Counter()
    token_counter: TypeCounter[str] = Counter()
    node_counter: TypeCounter[str] = Counter()
    type_counter: TypeCounter[str] = Counter()
    dataset_config = config.dataset
    train_data_path = path.join(dataset_directory, f"{dataset_config.name}.{config.train_holdout}.c2s")
    with_types = "type" in dataset_config
    with open(train_data_path, "r") as train_file:
        for line in tqdm(train_file, total=count_lines_in_file(train_data_path)):
            label, *path_contexts = line.split()
            target_counter.update(parse_token(label, False))
            for path_context in path_contexts:
                if with_types:
                    from_type, from_token, path_nodes, to_token, to_type = path_context.split(",")
                    type_counter.update(parse_token(from_type, dataset_config.type.is_splitted))
                    type_counter.update(parse_token(to_type, dataset_config.type.is_splitted))
                else:
                    from_token, path_nodes, to_token = path_context.split(",")
                token_counter.update(parse_token(from_token, dataset_config.token.is_splitted))
                token_counter.update(parse_token(to_token, dataset_config.token.is_splitted))
                node_counter.update(path_nodes.split("|"))
    return _counters_to_vocab(dataset_config, token_counter, target_counter, node_counter, type_counter)


def convert_vocabulary(config: DictConfig, original_vocabulary_path: str) -> Vocabulary:
    with open(original_vocabulary_path, "rb") as dict_file:
        subtoken_to_count: TypeCounter[str] = Counter(pickle.load(dict_file))
        node_to_count: TypeCounter[str] = Counter(pickle.load(dict_file))
        target_to_count: TypeCounter[str] = Counter(pickle.load(dict_file))
    return _counters_to_vocab(config, subtoken_to_count, target_to_count, node_to_count, Counter())


def preprocess(config: DictConfig):
    dataset_directory = join(config.data_folder, config.dataset.name)
    possible_dict = join(dataset_directory, f"{config.dataset.name}.dict.c2s")
    if exists(possible_dict):
        vocabulary = convert_vocabulary(config, possible_dict)
    else:
        vocabulary = collect_vocabulary(config, dataset_directory)
    vocabulary.dump_vocabulary(join(dataset_directory, "vocabulary.pkl"))


if __name__ == "__main__":
    with initialize_config_dir(get_config_directory()):
        _config = compose("main", overrides=argv[1:])
        preprocess(_config)
