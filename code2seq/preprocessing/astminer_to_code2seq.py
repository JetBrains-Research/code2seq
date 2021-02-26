from argparse import ArgumentParser
from os import path, remove
from typing import Dict

import numpy
from tqdm import tqdm

from code2seq.utils.filesystem import count_lines_in_file
from random import shuffle, seed


def _get_id2value_from_csv(path_: str) -> Dict[str, str]:
    return dict(numpy.genfromtxt(path_, delimiter=",", dtype=(str, str))[1:])


def preprocess_csv(data_folder: str, dataset_name: str, holdout_name: str, is_shuffled: bool):
    """
    Preprocessing for files tokens.csv, paths.csv, node_types.csv
    """
    dataset_path = path.join(data_folder, dataset_name)
    id_to_token_data_path = path.join(dataset_path, f"tokens.{holdout_name}.csv")
    id_to_type_data_path = path.join(dataset_path, f"node_types.{holdout_name}.csv")
    id_to_paths_data_path = path.join(dataset_path, f"paths.{holdout_name}.csv")
    path_contexts_path = path.join(dataset_path, f"path_contexts.{holdout_name}.csv")
    output_c2s_path = path.join(dataset_path, f"{dataset_name}.{holdout_name}.c2s")

    id_to_paths_stored = _get_id2value_from_csv(id_to_paths_data_path)
    id_to_paths = {index: [n for n in nodes.split()] for index, nodes in id_to_paths_stored.items()}

    id_to_node_types = _get_id2value_from_csv(id_to_type_data_path)
    id_to_node_types = {index: node_type.rsplit(" ", maxsplit=1)[0] for index, node_type in id_to_node_types.items()}

    id_to_tokens = _get_id2value_from_csv(id_to_token_data_path)

    if path.exists(output_c2s_path):
        remove(output_c2s_path)
    with open(path_contexts_path, "r") as path_contexts_file, open(output_c2s_path, "a+") as c2s_output:
        output_lines = []
        for line in tqdm(path_contexts_file, total=count_lines_in_file(path_contexts_path)):
            label, *path_contexts = line.split()
            parsed_line = [label]
            for path_context in path_contexts:
                from_token_id, path_types_id, to_token_id = path_context.split(",")
                from_token, to_token = id_to_tokens[from_token_id], id_to_tokens[to_token_id]
                nodes = [id_to_node_types[p_] for p_ in id_to_paths[path_types_id]]
                parsed_line.append(",".join([from_token, "|".join(nodes), to_token]))
            output_lines.append(" ".join(parsed_line + ["\n"]))
        if is_shuffled:
            shuffle(output_lines)
        c2s_output.write("".join(output_lines))


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("data", type=str)
    arg_parser.add_argument("--shuffle", action="store_true")
    args = arg_parser.parse_args()
    data_path = path.join("data", args.data)

    seed(7)
    for holdout in ["train", "val", "test"]:
        print(f"preprocessing {holdout} data")
        preprocess_csv("data", args.data, holdout, args.shuffle)
