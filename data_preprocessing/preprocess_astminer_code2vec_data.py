from argparse import ArgumentParser
from collections import Counter
from os import path
from typing import Tuple, List, Dict
from multiprocessing import cpu_count

from tqdm import tqdm
import numpy
import pickle

from configs import get_preprocessing_config_code2seq_params, PreprocessingConfig
from dataset import Vocabulary
from utils.common import UNK, count_lines_in_file

from utils.preprocessing import convert_holdout, DATA_FOLDER, vocab_from_counters


def _get_id2value_from_csv(data_path: str) -> Dict[int, str]:
    data = numpy.genfromtxt(data_path, delimiter=",", dtype=(str, str))[1:]
    return {int(index): values for index, values in data}


def preprocess_csv(data_path: str, holdout_name: str):
    """
    Preprocessing for files tokens.csv, paths.csv, node_types.csv
    """
    token_data_path = path.join(data_path, f"tokens.{holdout_name}.csv")
    type_data_path = path.join(data_path, f"node_types.{holdout_name}.csv")
    paths_data_path = path.join(data_path, f"paths.{holdout_name}.csv")

    paths = _get_id2value_from_csv(paths_data_path)
    paths = {index: [int(n) for n in nodes.split()] for index, nodes in paths.items()}

    node_types = _get_id2value_from_csv(type_data_path)
    node_types = {index: node_type.rsplit(" ", maxsplit=1)[0] for index, node_type in node_types.items()}

    tokens = _get_id2value_from_csv(token_data_path)
    tokens = {index: token_seq.split("|") for index, token_seq in tokens.items()}

    return tokens, node_types, paths


def collect_vocabulary(
        config: PreprocessingConfig,
        tokens: Dict[int, List[int]],
        node_types: Dict[int, List[int]],
        paths: Dict[int, List[int]]
) -> Vocabulary:
    train_contexts_path = path.join(DATA_FOLDER, config.dataset_name, "path_contexts.train.csv")
    target_counter = Counter()
    token_counter = Counter()
    type_counter = Counter()

    with open(train_contexts_path, "r") as train_contexts_file:
        for line in tqdm(train_contexts_file, total=count_lines_in_file(train_contexts_path)):
            label, *path_contexts = line.split()
            target_counter.update([label])
            cur_tokens = []
            cur_types = []
            for pc in path_contexts:
                from_token_id, path_id, to_token_id = (int(_) for _ in pc.split(","))
                # Extracting paths
                cur_types += [node_types[_id] for _id in paths[path_id]]

                cur_tokens += tokens[from_token_id] + tokens[to_token_id]
            token_counter.update(cur_tokens)
            type_counter.update(cur_types)

    return vocab_from_counters(config, token_counter, target_counter, type_counter)


def convert_path_context_to_ids(
    path_context: str,
    vocab: Vocabulary,
    paths: Dict[int, List[int]],
    node_types: Dict[int, List[int]],
    tokens: Dict[int, List[int]]
) -> Tuple[List[int], List[int], List[int]]:
    from_token_id, path_types_id, to_token_id = (int(_) for _ in path_context.split(","))
    nodes = paths[path_types_id]
    type_unk = vocab.type_to_id[UNK]

    from_tokens, to_tokens = tokens[from_token_id], tokens[to_token_id]
    token_unk = vocab.token_to_id[UNK]

    return (
        [vocab.token_to_id.get(_t, token_unk) for _t in from_tokens],
        [vocab.type_to_id.get(node_types[_n], type_unk) for _n in nodes],
        [vocab.token_to_id.get(_t, token_unk) for _t in to_tokens],
    )


def preprocess(config: PreprocessingConfig, n_jobs: int):
    # Collect vocabulary from train holdout if needed
    data_path = path.join(DATA_FOLDER, config.dataset_name)
    vocab_path = path.join(data_path, "vocabulary.pkl")

    if path.exists(vocab_path):
        vocab = Vocabulary.load(vocab_path)
    else:
        tokens, node_types, paths = preprocess_csv(data_path, "train")
        vocab = collect_vocabulary(config, tokens=tokens, paths=paths, node_types=node_types)
        vocab.dump(vocab_path)

    for holdout_name in "train", "test", "val":
        tokens, node_types, paths = preprocess_csv(data_path, holdout_name)
        convert_holdout(
            data_path,
            holdout_name,
            vocab,
            config,
            n_jobs,
            convert_path_context_to_ids,
            paths=paths,
            tokens=tokens,
            node_types=node_types,
        )


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("data", type=str)
    arg_parser.add_argument("--n-jobs", type=int, default=None)
    args = arg_parser.parse_args()

    preprocess(
        get_preprocessing_config_code2seq_params(args.data), args.n_jobs or cpu_count(),
    )
