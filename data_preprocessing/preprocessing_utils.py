import subprocess
from collections import Counter
from os import mkdir
from os.path import exists
from shutil import rmtree
from typing import List, Generator

from configs import PreprocessingConfig
from dataset import Vocabulary
from utils.common import SOS, EOS, PAD, UNK

SEPARATOR = "|"
SEED = 7
DATA_FOLDER = "data"
TRAIN_HOLDOUT = "train"
VAL_HOLDOUT = "val"
TEST_HOLDOUT = "test"
HOLDOUTS = [TRAIN_HOLDOUT, VAL_HOLDOUT, TEST_HOLDOUT]


def parse_token(token: str, is_split: bool) -> List[str]:
    return token.split(SEPARATOR) if is_split else [token]


def vocab_from_counters(
    config: PreprocessingConfig, token_counter: Counter, target_counter: Counter, type_counter: Counter
) -> Vocabulary:
    vocab = Vocabulary()
    names_additional_tokens = [SOS, EOS, PAD, UNK] if config.wrap_name else [PAD, UNK]
    vocab.add_from_counter("token_to_id", token_counter, config.subtoken_vocab_max_size, names_additional_tokens)
    target_additional_tokens = [SOS, EOS, PAD, UNK] if config.wrap_target else [PAD, UNK]
    vocab.add_from_counter("label_to_id", target_counter, config.target_vocab_max_size, target_additional_tokens)
    paths_additional_tokens = [SOS, EOS, PAD, UNK] if config.wrap_path else [PAD, UNK]
    vocab.add_from_counter("type_to_id", type_counter, -1, paths_additional_tokens)
    return vocab


def create_folder(path: str, is_clean: bool = True) -> None:
    if is_clean and exists(path):
        rmtree(path)
    if not exists(path):
        mkdir(path)


def read_file_by_batch(filepath: str, batch_size: int) -> Generator[List[str], None, None]:
    with open(filepath, "r") as file:
        lines = []
        for line in file:
            lines.append(line.strip())
            if len(lines) == batch_size:
                yield lines
                lines = []
    yield lines


def count_lines_in_file(file_path: str) -> int:
    command_result = subprocess.run(["wc", "-l", file_path], capture_output=True, encoding="utf-8")
    if command_result.returncode != 0:
        raise RuntimeError(f"Counting lines in {file_path} failed with error\n{command_result.stderr}")
    return int(command_result.stdout.split()[0])
