import pickle
from argparse import ArgumentParser
from collections import Counter
from math import ceil
from multiprocessing import Pool, cpu_count
from os import path
from typing import Tuple, List, Generator

from tqdm import tqdm

from configs import get_preprocessing_config_code2seq_params, PreprocessingConfig
from dataset import Vocabulary, BufferedPathContext, ConvertParameters
from utils.common import SOS, EOS, PAD, UNK, count_lines_in_file, create_folder, FROM_TOKEN, TO_TOKEN, PATH_TYPES

DATA_FOLDER = "data"
DESCRIPTION_FILE = "description.csv"
BUFFERED_PATH_TEMPLATE = "buffered_paths_{}.pkl"
SEPARATOR = "|"


def _vocab_from_counters(
    config: PreprocessingConfig, token_counter: Counter, target_counter: Counter, type_counter: Counter
) -> Vocabulary:
    vocab = Vocabulary()
    vocab.add_from_counter(
        "token_to_id", token_counter, config.subtoken_vocab_max_size, [SOS, EOS, PAD, UNK],
    )
    vocab.add_from_counter("label_to_id", target_counter, config.target_vocab_max_size, [SOS, EOS, PAD, UNK])
    vocab.add_from_counter("type_to_id", type_counter, -1, [SOS, EOS, PAD, UNK])
    return vocab


def _parse_token(token: str, is_split: bool) -> List[str]:
    return token.split(SEPARATOR) if is_split else [token]


def collect_vocabulary(config: PreprocessingConfig) -> Vocabulary:
    target_counter = Counter()
    token_counter = Counter()
    type_counter = Counter()
    train_data_path = path.join(DATA_FOLDER, config.dataset_name, f"{config.dataset_name}.train.c2s")
    with open(train_data_path, "r") as train_file:
        for line in tqdm(train_file, total=count_lines_in_file(train_data_path)):
            label, *path_contexts = line.split()
            target_counter.update(_parse_token(label, config.split_target))
            cur_tokens = []
            cur_types = []
            for path_context in path_contexts:
                from_token, path_types, to_token = path_context.split(",")
                cur_tokens += _parse_token(from_token, config.split_names) + _parse_token(to_token, config.split_names)
                cur_types += path_types.split("|")
            token_counter.update(cur_tokens)
            type_counter.update(cur_types)
    return _vocab_from_counters(config, token_counter, target_counter, type_counter)


def convert_vocabulary(config: PreprocessingConfig) -> Vocabulary:
    with open(path.join(DATA_FOLDER, config.dataset_name, f"{config.dataset_name}.dict.c2s"), "rb") as dict_file:
        subtoken_to_count = Counter(pickle.load(dict_file))
        node_to_count = Counter(pickle.load(dict_file))
        target_to_count = Counter(pickle.load(dict_file))
    return _vocab_from_counters(config, subtoken_to_count, target_to_count, node_to_count)


def _convert_path_context_to_ids(
    is_split: bool, path_context: str, vocab: Vocabulary
) -> Tuple[List[int], List[int], List[int]]:
    from_token, path_types, to_token = path_context.split(",")

    from_token = _parse_token(from_token, is_split)
    to_token = _parse_token(to_token, is_split)

    token_unk = vocab.token_to_id[UNK]
    type_unk = vocab.type_to_id[UNK]
    return (
        [vocab.token_to_id.get(_t, token_unk) for _t in from_token],
        [vocab.type_to_id.get(_t, type_unk) for _t in path_types.split("|")],
        [vocab.token_to_id.get(_t, token_unk) for _t in to_token],
    )


def _convert_raw_buffer(convert_args: Tuple[List[str], PreprocessingConfig, Vocabulary, str, int]):
    lines, config, vocab, output_folder, buffer_id = convert_args
    labels, from_tokens, path_types, to_tokens = [], [], [], []
    for line in lines:
        label, *path_contexts = line.split()
        label = _parse_token(label, config.split_target)
        labels.append([vocab.label_to_id.get(_l, vocab.label_to_id[UNK]) for _l in label])
        converted_context = [_convert_path_context_to_ids(config.split_names, pc, vocab) for pc in path_contexts]
        from_tokens.append([cc[0] for cc in converted_context])
        path_types.append([cc[1] for cc in converted_context])
        to_tokens.append([cc[2] for cc in converted_context])

    bpc = BufferedPathContext.create_from_lists(
        (labels, ConvertParameters(config.max_target_parts, config.wrap_target, vocab.label_to_id)),
        {
            FROM_TOKEN: (from_tokens, ConvertParameters(config.max_name_parts, config.wrap_name, vocab.token_to_id),),
            PATH_TYPES: (path_types, ConvertParameters(config.max_path_length, config.wrap_path, vocab.type_to_id)),
            TO_TOKEN: (to_tokens, ConvertParameters(config.max_name_parts, config.wrap_name, vocab.token_to_id)),
        },
    )

    with open(path.join(output_folder, DESCRIPTION_FILE), "a") as desc_file:
        n_samples = len(bpc.contexts_per_label)
        n_paths = sum(bpc.contexts_per_label)
        desc_file.write(f"{buffer_id},{BUFFERED_PATH_TEMPLATE.format(buffer_id)},{n_samples},{n_paths}\n")
    bpc.dump(path.join(output_folder, BUFFERED_PATH_TEMPLATE.format(buffer_id)))


def _read_file_by_batch(filepath: str, batch_size: int) -> Generator[List[str], None, None]:
    with open(filepath, "r") as file:
        lines = []
        for line in file:
            lines.append(line.strip())
            if len(lines) == batch_size:
                yield lines
                lines = []
    yield lines


def convert_holdout(holdout_name: str, vocab: Vocabulary, config: PreprocessingConfig, n_jobs: int):
    holdout_data_path = path.join(DATA_FOLDER, config.dataset_name, f"{config.dataset_name}.{holdout_name}.c2s")
    holdout_output_folder = path.join(DATA_FOLDER, config.dataset_name, holdout_name)
    create_folder(holdout_output_folder)
    with open(path.join(holdout_output_folder, DESCRIPTION_FILE), "w") as desc_file:
        desc_file.write("id,filename,n_samples,n_paths\n")
    with Pool(n_jobs) as pool:
        results = pool.imap(
            _convert_raw_buffer,
            (
                (lines, config, vocab, holdout_output_folder, pos)
                for pos, lines in enumerate(_read_file_by_batch(holdout_data_path, config.buffer_size))
            ),
        )
        n_buffers = ceil(count_lines_in_file(holdout_data_path) / config.buffer_size)
        _ = [_ for _ in tqdm(results, total=n_buffers)]


def preprocess(config: PreprocessingConfig, is_vocab_collected: bool, n_jobs: int):
    # Collect vocabulary from train holdout if needed
    vocab_path = path.join(DATA_FOLDER, config.dataset_name, "vocabulary.pkl")
    if path.exists(vocab_path):
        vocab = Vocabulary.load(vocab_path)
    else:
        vocab = collect_vocabulary(config) if is_vocab_collected else convert_vocabulary(config)
        vocab.dump(vocab_path)
    for holdout in ["train", "val", "test"]:
        convert_holdout(holdout, vocab, config, n_jobs)


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("data", type=str)
    arg_parser.add_argument("--collect-vocabulary", action="store_true")
    arg_parser.add_argument("--n-jobs", type=int, default=None)
    args = arg_parser.parse_args()

    preprocess(get_preprocessing_config_code2seq_params(args.data), args.collect_vocabulary, args.n_jobs or cpu_count())
