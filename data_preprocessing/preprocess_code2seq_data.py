import pickle
from argparse import ArgumentParser
from collections import Counter
from os import path
from typing import Tuple, List

from tqdm import tqdm

from configs import get_preprocessing_config_code2seq_params, PreprocessingConfig
from dataset import Vocabulary, create_standard_bpc
from utils.common import SOS, EOS, PAD, UNK, count_lines_in_file, create_folder


def collect_vocabulary(config: PreprocessingConfig) -> Vocabulary:
    label_counter = Counter()
    token_counter = Counter()
    type_counter = Counter()
    train_data_path = path.join(config.data_path, f"{config.dataset_name}.train.c2s")
    with open(train_data_path, "r") as train_file:
        for line in tqdm(train_file, total=count_lines_in_file(train_data_path)):
            label, *path_contexts = line.split()
            label_counter.update(label.split("|"))
            cur_tokens = []
            cur_types = []
            for path_context in path_contexts:
                from_token, path_types, to_token = path_context.split(",")
                cur_tokens += from_token.split("|") + to_token.split("|")
                cur_types += path_types.split("|")
            token_counter.update(cur_tokens)
            type_counter.update(cur_types)
    vocab = Vocabulary()
    vocab.add_from_counter(
        "token_to_id", token_counter, config.subtoken_vocab_max_size, [SOS, EOS, PAD, UNK],
    )
    vocab.add_from_counter("label_to_id", label_counter, config.target_vocab_max_size, [SOS, EOS, PAD, UNK])
    vocab.add_from_counter("type_to_id", type_counter, -1, [SOS, EOS, PAD, UNK])
    return vocab


def _convert_path_context_to_ids(path_context: str, vocab: Vocabulary) -> Tuple[List[int], List[int], List[int]]:
    from_token, path_types, to_token = path_context.split(",")
    token_unk = vocab.token_to_id[UNK]
    type_unk = vocab.type_to_id[UNK]
    return (
        [vocab.token_to_id.get(_t, token_unk) for _t in from_token.split("|")],
        [vocab.type_to_id.get(_t, type_unk) for _t in path.split("|")],
        [vocab.token_to_id.get(_t, token_unk) for _t in to_token.split("|")],
    )


def convert_holdout(holdout_name: str, vocab: Vocabulary, config: PreprocessingConfig):
    holdout_data_path = path.join(config.data_path, f"{config.dataset_name}.{holdout_name}.c2s")
    holdout_output_folder = path.join(config.data_path, holdout_name)
    create_folder(holdout_output_folder)
    label_unk = vocab.label_to_id[UNK]
    with open(holdout_data_path, "r") as holdout_file:
        labels, from_tokens, path_types, to_tokens = [], [], [], []
        for i, line in tqdm(enumerate(holdout_file), total=count_lines_in_file(holdout_data_path)):
            label, *path_contexts = line.split()
            labels.append([vocab.label_to_id.get(_l, label_unk) for _l in label.split("|")])
            from_tokens_ids, path_types_ids, to_tokens_ids = list(
                zip(*[_convert_path_context_to_ids(pc, vocab) for pc in path_contexts])
            )
            from_tokens.append(from_tokens_ids)
            path_types.append(path_types_ids)
            to_tokens.append(to_tokens_ids)

            if len(labels) == config.buffer_size:
                buffered_path_context = create_standard_bpc(config, vocab, labels, from_tokens, path_types, to_tokens)
                buffered_path_context.dump(
                    path.join(holdout_output_folder, f"buffered_paths_{i // config.buffer_size}.pkl")
                )
                labels, from_tokens, path_types, to_tokens = [], [], [], []
        if len(labels) > 0:
            buffered_path_context = create_standard_bpc(config, vocab, labels, from_tokens, path_types, to_tokens)
            buffered_path_context.dump(
                path.join(holdout_output_folder, f"buffered_paths_{i // config.buffer_size}.pkl")
            )


def preprocess(config: PreprocessingConfig):
    # Collect vocabulary from train holdout if needed
    vocab_path = path.join(config.data_path, "vocabulary.pkl")
    if path.exists(vocab_path):
        with open(vocab_path, "rb") as vocab_file:
            vocab = pickle.load(vocab_file)
    else:
        vocab = collect_vocabulary(config)
        with open(vocab_path, "wb") as vocab_file:
            pickle.dump(vocab, vocab_file)
    convert_holdout("train", vocab, config)
    convert_holdout("val", vocab, config)
    convert_holdout("test", vocab, config)


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("data", type=str)
    args = arg_parser.parse_args()

    preprocess(get_preprocessing_config_code2seq_params(args.data))
