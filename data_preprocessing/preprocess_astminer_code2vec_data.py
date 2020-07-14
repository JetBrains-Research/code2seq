from argparse import ArgumentParser
from collections import Counter, defaultdict
from os import path
from typing import Tuple, List, Dict, Any
from multiprocessing import cpu_count

from tqdm import tqdm
import pandas as pd
import pickle
from operator import itemgetter
import os

from configs import (
    get_preprocessing_config_astminer_code2vec_params,
    PreprocessingConfig,
)
from dataset import Vocabulary
from utils.common import UNK, count_lines_in_file, vocab_from_counters

from data_preprocessing.buffer_utils import convert_holdout, DATA_FOLDER


def split_data(config: PreprocessingConfig) -> None:
    data_path = path.join(DATA_FOLDER, config.dataset_name, "c")
    paths_contexts_path = path.join(data_path, "path_contexts.csv")

    # split paths_contexts into train test and validation
    with open(paths_contexts_path, "r") as paths:
        # last line in the file contains line breaker
        data = paths.read().split("\n")[:-2]
        label_to_line = defaultdict(list)
        for line in data:
            label, *_ = line.split(" ")
            label_to_line[label].append(line)

        for label, lines in label_to_line.items():
            num_lines = len(lines)
            train = round(num_lines * args.train)
            test = round(num_lines * args.test)

            for holdout_name, chunk in zip(
                ("train", "test", "val"),
                (lines[:train], lines[train : train + test], lines[train + test :]),
            ):
                file = path.join(
                    DATA_FOLDER, config.dataset_name, f"{holdout_name}.csv"
                )

                with open(file, "a+") as holdout_file:
                    holdout_file.write("\n".path.join(chunk + [""]))
    # os.rmdir(data_path)


def _get_id2value_from_csv(data_path: str) -> Dict[str, Dict[int, str]]:
    return pd.read_csv(data_path).fillna("").set_index("id").to_dict()


def _dump_dict(path: str, data: Any) -> None:
    with open(path, "wb+") as file:
        pickle.dump(data, file)


def _load_dict(path: str) -> Any:
    with open(path, "rb") as file:
        data = pickle.load(file)
    return data


def preprocess_csv(
    config: PreprocessingConfig,
) -> Tuple[Dict[int, List[int]], Dict[int, str], Dict[int, List[int]]]:
    """
    Preprocessing for files tokens.csv, paths.csv, node_types.csv
    """
    data_path = path.join(DATA_FOLDER, config.dataset_name)
    token_data_path = path.join(data_path, "c", "tokens.csv")
    type_data_path = path.join(data_path, "c", "node_types.csv")
    paths_data_path = path.join(data_path, "c", "paths.csv")

    paths = _get_id2value_from_csv(paths_data_path)["path"]
    paths = {index: list(map(int, nodes.split())) for index, nodes in paths.items()}

    node_types = _get_id2value_from_csv(type_data_path)["node_type"]

    tokens = _get_id2value_from_csv(token_data_path)["token"]
    tokens = {index: token_seq.split("|") for index, token_seq in tokens.items()}

    _dump_dict(path.join(data_path, "tokens.pkl"), tokens)
    _dump_dict(path.join(data_path, "node_types.pkl"), node_types)
    _dump_dict(path.join(data_path, "paths.pkl"), paths)
    return paths, node_types, tokens


def collect_vocabulary(
    config: PreprocessingConfig,
    paths: Dict[int, List[int]],
    node_types: Dict[int, str],
    tokens: Dict[int, List[int]],
) -> Vocabulary:
    train_contexts_path = path.join(DATA_FOLDER, config.dataset_name, "train.csv")
    target_counter = Counter()
    token_counter = Counter()
    type_counter = Counter()

    with open(train_contexts_path, "r") as train_contexts_file:
        for line in tqdm(
            train_contexts_file, total=count_lines_in_file(train_contexts_path)
        ):
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
    path_context: str,
    vocab: Vocabulary,
    paths: Dict[int, List[int]],
    tokens: Dict[int, List[int]],
) -> Tuple[List[int], List[int], List[int]]:
    from_token_id, path_types_id, to_token_id = list(map(int, path_context.split(",")))

    nodes = paths[path_types_id]
    type_unk = vocab.type_to_id[UNK]

    from_tokens, to_tokens = tokens[from_token_id], tokens[to_token_id]
    token_unk = vocab.token_to_id[UNK]

    return (
        [vocab.token_to_id.get(_t, token_unk) for _t in from_tokens],
        [vocab.type_to_id.get(_n, type_unk) for _n in nodes],
        [vocab.token_to_id.get(_t, token_unk) for _t in to_tokens],
    )


def _split_context(
    line: str, vocab: Vocabulary, **kwargs: Any
) -> Tuple[List[int], List[Tuple[List[int], List[int], List[int]]]]:
    paths, tokens = kwargs["paths"], kwargs["tokens"]
    label, *path_contexts = line.split()
    converted_context = [
        _convert_path_context_to_ids(pc, vocab, paths, tokens) for pc in path_contexts
    ]
    return [vocab.label_to_id[label]], converted_context


def preprocess(config: PreprocessingConfig, n_jobs: int):
    # Collect vocabulary from train holdout if needed
    data_path = path.join(DATA_FOLDER, config.dataset_name)
    if not all(
        f"{holdout_name}.csv" in os.listdir(data_path)
        for holdout_name in ("train", "test", "val")
    ):
        split_data(config)

    if not all(
        f"{csv_name}.pkl" in os.listdir(data_path)
        for csv_name in ("tokens", "paths", "node_types")
    ):
        paths, node_types, tokens = preprocess_csv(config)
    else:
        tokens = _load_dict(path.join(data_path, "tokens.pkl"))
        paths = _load_dict(path.join(data_path, "paths.pkl"))
        node_types = _load_dict(path.join(data_path, "node_types.pkl"))

    vocab_path = path.join(data_path, "vocabulary.pkl")

    if path.exists(vocab_path):
        vocab = Vocabulary.load(vocab_path)
    else:
        vocab = collect_vocabulary(config, paths, node_types, tokens)
        vocab.dump(vocab_path)

    for holdout_name in "train", "test", "val":
        holdout_data_path = path.join(
            DATA_FOLDER, config.dataset_name, f"{holdout_name}.csv"
        )
        holdout_output_folder = path.join(
            DATA_FOLDER, config.dataset_name, holdout_name
        )
        convert_holdout(
            holdout_data_path,
            holdout_output_folder,
            vocab,
            config,
            n_jobs,
            _split_context,
            paths=paths,
            tokens=tokens,
        )


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("data", type=str)
    arg_parser.add_argument("--collect-vocabulary", action="store_true")
    arg_parser.add_argument("--n-jobs", type=int, default=None)
    arg_parser.add_argument("--train", type=float, default=0.7)
    arg_parser.add_argument("--test", type=float, default=0.1)
    arg_parser.add_argument("--val", type=float, default=0.2)
    args = arg_parser.parse_args()

    if args.train + args.test + args.val != 1.0:
        raise ValueError("Incorrect train/test/val split")

    preprocess(
        get_preprocessing_config_astminer_code2vec_params(
            args.data, args.train, args.test, args.val
        ),
        args.n_jobs or cpu_count(),
    )
