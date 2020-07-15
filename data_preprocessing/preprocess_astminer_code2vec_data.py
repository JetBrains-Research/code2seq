from argparse import ArgumentParser
from collections import Counter
from os import path
from typing import Tuple, List, Dict, Any
from multiprocessing import cpu_count

from tqdm import tqdm
import pandas as pd
import pickle
from operator import itemgetter
import os

from configs import (
    get_preprocessing_config_code2seq_params,
    PreprocessingConfig,
)
from dataset import Vocabulary
from utils.common import UNK, count_lines_in_file, vocab_from_counters

from data_preprocessing.buffer_utils import convert_holdout, DATA_FOLDER


def _get_id2value_from_csv(data_path: str) -> Dict[str, Dict[int, str]]:
    return pd.read_csv(data_path).fillna("").set_index("id").to_dict()


def preprocess_csv(data_path: str, holdout_name: str):
    """
    Preprocessing for files tokens.csv, paths.csv, node_types.csv
    """
    token_data_path = path.join(data_path, f"tokens.{holdout_name}.csv")
    type_data_path = path.join(data_path, f"node_types.{holdout_name}.csv")
    paths_data_path = path.join(data_path, f"paths.{holdout_name}.csv")

    paths = _get_id2value_from_csv(paths_data_path)["path"]
    paths = {index: list(map(int, nodes.split())) for index, nodes in paths.items()}

    node_types = _get_id2value_from_csv(type_data_path)["node_type"]

    tokens = _get_id2value_from_csv(token_data_path)["token"]
    tokens = {index: token_seq.split("|") for index, token_seq in tokens.items()}

    parsed_data_path = path.join(data_path, f"{holdout_name}.pkl")
    with open(parsed_data_path, "wb+") as parsed_data_file:
        pickle.dump({"tokens": tokens, "node_types": node_types, "paths": paths}, parsed_data_file)


def _load_preprocessed_data(
    train_parsed_path: str,
) -> Tuple[Dict[int, List[int]], Dict[int, List[int]], Dict[int, List[int]]]:
    with open(train_parsed_path, "rb") as train_parsed_data:
        data = pickle.load(train_parsed_data)
        return data["tokens"], data["node_types"], data["paths"]


def collect_vocabulary(config: PreprocessingConfig) -> Vocabulary:
    train_parsed_path = path.join(DATA_FOLDER, config.dataset_name, "train.pkl")
    train_contexts_path = path.join(DATA_FOLDER, config.dataset_name, "path_contexts.train.csv")
    target_counter = Counter()
    token_counter = Counter()
    type_counter = Counter()

    tokens, node_types, paths = _load_preprocessed_data(train_parsed_path)

    with open(train_contexts_path, "r") as train_contexts_file:
        for line in tqdm(train_contexts_file, total=count_lines_in_file(train_contexts_path)):
            label, *path_contexts = line.split()
            target_counter.update([label])
            cur_tokens = []
            cur_types = []
            for pc in path_contexts:
                from_token_id, path_id, to_token_id = list(map(int, pc.split(",")))
                # Extracting paths
                path_types_itemgetter = itemgetter(*paths[path_id])
                path_types = list(path_types_itemgetter(node_types))
                cur_types += path_types

                from_tokens, to_tokens = tokens[from_token_id], tokens[to_token_id]
                cur_tokens += from_tokens + to_tokens
            token_counter.update(cur_tokens)
            type_counter.update(cur_types)

    return vocab_from_counters(config, token_counter, target_counter, type_counter)


def _convert_path_context_to_ids(
    path_context: str, vocab: Vocabulary, **kwargs
) -> Tuple[List[int], List[int], List[int]]:
    paths, node_types, tokens = kwargs["paths"], kwargs["node_types"], kwargs["tokens"]
    from_token_id, path_types_id, to_token_id = list(map(int, path_context.split(",")))
    nodes = paths[path_types_id]
    type_unk = vocab.type_to_id[UNK]

    from_tokens, to_tokens = tokens[from_token_id], tokens[to_token_id]
    token_unk = vocab.token_to_id[UNK]

    return (
        [vocab.token_to_id.get(_t, token_unk) for _t in from_tokens],
        [vocab.type_to_id.get(node_types[_n], type_unk) for _n in nodes],
        [vocab.token_to_id.get(_t, token_unk) for _t in to_tokens],
    )


def split_context(
    line: str, vocab: Vocabulary, **kwargs
) -> Tuple[List[int], List[Tuple[List[int], List[int], List[int]]]]:
    label, *path_contexts = line.split()
    converted_context = [_convert_path_context_to_ids(pc, vocab, **kwargs) for pc in path_contexts]
    return [vocab.label_to_id[label]], converted_context


def preprocess(config: PreprocessingConfig, n_jobs: int):
    # Collect vocabulary from train holdout if needed
    data_path = path.join(DATA_FOLDER, config.dataset_name)

    for folder_name in ("train", "test", "val"):
        if f"{folder_name}.pkl" not in os.listdir(data_path):
            data_path = path.join(DATA_FOLDER, config.dataset_name)
            preprocess_csv(data_path, folder_name)

    vocab_path = path.join(data_path, "vocabulary.pkl")

    if path.exists(vocab_path):
        vocab = Vocabulary.load(vocab_path)
    else:
        vocab = collect_vocabulary(config)
        vocab.dump(vocab_path)

    for holdout_name in "train", "test", "val":
        holdout_data_path = path.join(data_path, f"path_contexts.{holdout_name}.csv")
        holdout_output_folder = path.join(data_path, holdout_name)
        holdout_parsed_path = path.join(data_path, f"{holdout_name}.pkl")
        tokens, node_types, paths = _load_preprocessed_data(holdout_parsed_path)
        convert_holdout(
            holdout_data_path, holdout_output_folder, vocab, config, n_jobs, split_context,
            paths=paths, tokens=tokens, node_types=node_types
        )


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("data", type=str)
    arg_parser.add_argument("--collect-vocabulary", action="store_true")
    arg_parser.add_argument("--n-jobs", type=int, default=None)
    args = arg_parser.parse_args()

    preprocess(
        get_preprocessing_config_code2seq_params(args.data), args.n_jobs or cpu_count(),
    )
