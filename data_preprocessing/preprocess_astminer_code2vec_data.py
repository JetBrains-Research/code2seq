from argparse import ArgumentParser
from collections import Counter
from os import path, remove
from typing import Tuple, List, Dict
from multiprocessing import cpu_count
from tqdm import tqdm

import numpy

from configs import get_preprocessing_config_code2seq_params

from data_preprocessing.preprocess_code2seq_data import preprocess, DATA_FOLDER
from utils.common import FROM_TOKEN, PATH_TYPES, TO_TOKEN, count_lines_in_file, create_folder, SOS, EOS, PAD, UNK


def _get_id2value_from_csv(data_path: str) -> Dict[str, str]:
    return dict(numpy.genfromtxt(data_path, delimiter=",", dtype=(str, str))[1:])


def preprocess_csv(data_folder: str, dataset_name: str, holdout_name: str):
    """
    Preprocessing for files tokens.csv, paths.csv, node_types.csv
    """
    data_path = path.join(data_folder, dataset_name)
    token_data_path = path.join(data_path, f"tokens.{holdout_name}.csv")
    type_data_path = path.join(data_path, f"node_types.{holdout_name}.csv")
    paths_data_path = path.join(data_path, f"paths.{holdout_name}.csv")
    path_contexts_path = path.join(data_path, f"path_contexts.{holdout_name}.csv")
    poj_c2s_path = path.join(data_path, f"{dataset_name}.{holdout_name}.c2s")

    id_to_paths = _get_id2value_from_csv(paths_data_path)
    id_to_paths = {index: [n for n in nodes.split()] for index, nodes in id_to_paths.items()}

    id_to_node_types = _get_id2value_from_csv(type_data_path)
    id_to_node_types = {index: node_type.rsplit(" ", maxsplit=1)[0] for index, node_type in id_to_node_types.items()}

    id_to_tokens = _get_id2value_from_csv(token_data_path)

    remove(poj_c2s_path)
    with open(path_contexts_path, "r") as path_contexts_file, open(poj_c2s_path, "a+") as output:
        for line in tqdm(path_contexts_file, total=count_lines_in_file(path_contexts_path)):
            label, *path_contexts = line.split()
            parsed_line = [label]
            for path_context in path_contexts:
                from_token_id, path_types_id, to_token_id = path_context.split(",")
                from_token, to_token = id_to_tokens[from_token_id], id_to_tokens[to_token_id]
                nodes = [id_to_node_types[p_] for p_ in id_to_paths[path_types_id]]
                parsed_line.append(",".join([from_token, "|".join(nodes), to_token]))
            output.write(" ".join(parsed_line))


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("data", type=str)
    arg_parser.add_argument("--n-jobs", type=int, default=None)
    args = arg_parser.parse_args()
    data_path = path.join(DATA_FOLDER, args.data)
    for holdout_name in "train", "test", "val":
        preprocess_csv(DATA_FOLDER, args.data, holdout_name)

    preprocess(get_preprocessing_config_code2seq_params(args.data), True, args.n_jobs or cpu_count())
