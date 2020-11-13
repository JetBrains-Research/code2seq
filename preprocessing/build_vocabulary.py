import pickle
from collections import Counter
from os import path
from os.path import join, exists
from typing import Counter as TypeCounter
from typing import List, Dict

from hydra import main as hydra_main
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
    config: DictConfig, token_counter: Counter, target_counter: Counter, node_counter: Counter, type_counter: Counter,
) -> Vocabulary:
    names_additional_tokens = [SOS, EOS, PAD, UNK] if config.wrap_name else [PAD, UNK]
    token_to_id = _counter_to_dict(token_counter, config.token_vocab_size, names_additional_tokens)
    target_additional_tokens = [SOS, EOS, PAD, UNK] if config.wrap_target else [PAD, UNK]
    label_to_id = _counter_to_dict(target_counter, config.target_vocab_size, target_additional_tokens)
    paths_additional_tokens = [SOS, EOS, PAD, UNK] if config.wrap_path else [PAD, UNK]
    node_to_id = _counter_to_dict(node_counter, None, paths_additional_tokens)

    vocabulary = Vocabulary(token_to_id, node_to_id, label_to_id)
    if len(type_counter) > 0:
        vocabulary.type_to_id = _counter_to_dict(type_counter)
    return vocabulary


def collect_vocabulary(config: DictConfig, dataset_name: str, with_types: bool = False) -> Vocabulary:
    target_counter: TypeCounter[str] = Counter()
    token_counter: TypeCounter[str] = Counter()
    node_counter: TypeCounter[str] = Counter()
    type_counter: TypeCounter[str] = Counter()
    train_data_path = path.join(config.data_folder, dataset_name, f"{dataset_name}.{config.train_holdout}.c2s")
    with open(train_data_path, "r") as train_file:
        for line in tqdm(train_file, total=count_lines_in_file(train_data_path)):
            label, *path_contexts = line.split()
            target_counter.update(parse_token(label, config.split_target))
            cur_tokens = []
            cur_nodes = []
            cur_types = []
            for path_context in path_contexts:
                if with_types:
                    from_type, from_token, path_nodes, to_token, to_type = path_context.split(",")
                    cur_types += parse_token(from_type, False)
                    cur_types += parse_token(to_type, False)
                else:
                    from_token, path_nodes, to_token = path_context.split(",")
                cur_tokens += parse_token(from_token, config.split_names)
                cur_tokens += parse_token(to_token, config.split_names)
                cur_nodes += path_nodes.split("|")
            token_counter.update(cur_tokens)
            node_counter.update(cur_nodes)
            type_counter.update(cur_types)
    return _counters_to_vocab(config, token_counter, target_counter, node_counter, type_counter)


def convert_vocabulary(config: DictConfig, original_vocabulary_path: str) -> Vocabulary:
    with open(original_vocabulary_path, "rb") as dict_file:
        subtoken_to_count: TypeCounter[str] = Counter(pickle.load(dict_file))
        node_to_count: TypeCounter[str] = Counter(pickle.load(dict_file))
        target_to_count: TypeCounter[str] = Counter(pickle.load(dict_file))
    return _counters_to_vocab(config, subtoken_to_count, target_to_count, node_to_count, Counter())


@hydra_main(config_path=get_config_directory(), config_name="main")
def preprocess(config: DictConfig):
    # TODO: build vocabulary if needed. Move logic to data modules.
    pass


if __name__ == "__main__":
    preprocess()
