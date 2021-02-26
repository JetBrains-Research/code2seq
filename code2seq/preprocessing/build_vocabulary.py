import pickle
from collections import Counter
from os.path import join, exists
from typing import Counter as TypeCounter, Any
from typing import List, Dict

import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from code2seq.utils.converting import parse_token
from code2seq.utils.filesystem import count_lines_in_file
from code2seq.utils.vocabulary import Vocabulary, SOS, EOS, PAD, UNK


def _counter_to_dict(values: Counter, n_most_common: int = None, additional_values: List[str] = None) -> Dict[str, int]:
    dict_values = []
    if additional_values is not None:
        dict_values += additional_values
        for value in additional_values:
            if value in values:
                del values[value]
    dict_values += list(zip(*values.most_common(n_most_common)))[0]
    return {value: i for i, value in enumerate(dict_values)}


def _counters_to_vocab(config: Dict, counters: Dict[str, TypeCounter[str]]) -> Vocabulary:
    additional_tokens = [SOS, EOS, PAD, UNK] if config["token"]["is_wrapped"] else [PAD, UNK]
    token_to_id = _counter_to_dict(counters["token"], config["token"]["vocabulary_size"], additional_tokens)
    additional_targets = [SOS, EOS, PAD, UNK] if config["target"]["is_wrapped"] else [PAD, UNK]
    label_to_id = _counter_to_dict(counters["target"], config["target"]["vocabulary_size"], additional_targets)
    additional_nodes = [SOS, EOS, PAD, UNK] if config["path"]["is_wrapped"] else [PAD, UNK]
    node_to_id = _counter_to_dict(counters["path"], config["path"]["vocabulary_size"], additional_nodes)

    vocabulary = Vocabulary(token_to_id, node_to_id, label_to_id)
    if "type" in counters:
        additional_types = [SOS, EOS, PAD, UNK] if config["type"]["is_wrapped"] else [PAD, UNK]
        vocabulary.type_to_id = _counter_to_dict(counters["type"], config["type"]["vocabulary_size"], additional_types)
    return vocabulary


def parse_path_context(config: Dict, path_context: List[str]) -> Dict[str, List[str]]:
    result = {}
    if len(path_context) == 5:
        from_type, from_token, path_nodes, to_token, to_type = path_context
        result["type"] = parse_token(from_type, config["type"]["is_splitted"]) + parse_token(
            to_type, config["type"]["is_splitted"]
        )
    else:
        from_token, path_nodes, to_token = path_context
    result["token"] = parse_token(from_token, config["token"]["is_splitted"]) + parse_token(
        to_token, config["token"]["is_splitted"]
    )
    result["path"] = parse_token(path_nodes, config["path"]["is_splitted"])
    return result


def collect_vocabulary(config: Dict[str, Any], train_holdout: str) -> Vocabulary:
    counters: Dict[str, TypeCounter[str]] = {k: Counter() for k in ["target", "token", "path", "type"] if k in config}
    with open(train_holdout, "r") as train_file:
        for line in tqdm(train_file, total=count_lines_in_file(train_holdout)):
            label, *path_contexts = line.split()
            counters["target"].update(parse_token(label, config["target"]["is_splitted"]))
            for path_context in path_contexts:
                parsed_context = parse_path_context(config, path_context.split(","))
                for key, value in parsed_context.items():
                    counters[key].update(value)
    return _counters_to_vocab(config, counters)


def convert_vocabulary(config: Dict, original_vocabulary_path: str) -> Vocabulary:
    counters: Dict[str, TypeCounter[str]] = {}
    with open(original_vocabulary_path, "rb") as dict_file:
        counters["token"] = Counter(pickle.load(dict_file))
        counters["path"] = Counter(pickle.load(dict_file))
        counters["target"] = Counter(pickle.load(dict_file))
    return _counters_to_vocab(config, counters)


@hydra.main(config_path="../configs", config_name="code2seq")
def preprocess(config: DictConfig):
    dataset_directory = join(config.data_folder, config.dataset.name)
    possible_dict = join(dataset_directory, f"{config.dataset.name}.dict.c2s")
    train_holdout = join(dataset_directory, f"{config.dataset.name}.{config.train_holdout}.c2s")
    dict_data_config = OmegaConf.to_container(config.dataset, True)
    if not isinstance(dict_data_config, dict):
        raise ValueError
    if exists(possible_dict):
        vocabulary = convert_vocabulary(dict_data_config, possible_dict)
    else:
        vocabulary = collect_vocabulary(dict_data_config, train_holdout)
    vocabulary.dump_vocabulary(join(dataset_directory, "vocabulary.pkl"))


if __name__ == "__main__":
    preprocess()
