import subprocess
from os import mkdir
from os.path import exists
from shutil import rmtree

from typing import List

import numpy

# sequence service tokens
SOS = "<SOS>"
EOS = "<EOS>"
PAD = "<PAD>"
UNK = "<UNK>"

# buffered path context dict keys
FROM_TOKEN = "from_token"
PATH_TYPES = "path_types"
TO_TOKEN = "to_token"


def count_lines_in_file(file_path: str) -> int:
    command_result = subprocess.run(["wc", "-l", file_path], capture_output=True, encoding="utf-8")
    if command_result.returncode != 0:
        raise RuntimeError(f"Counting lines in {file_path} failed with error\n{command_result.stderr}")
    return int(command_result.stdout.split()[0])


def create_folder(path: str, is_clean: bool = True) -> None:
    if is_clean and exists(path):
        rmtree(path)
    if not exists(path):
        mkdir(path)


def segment_sizes_to_slices(sizes: List) -> List:
    cum_sums = numpy.cumsum(sizes)
    start_of_segments = numpy.append([0], cum_sums[:-1])
    return [slice(start, end) for start, end in zip(start_of_segments, cum_sums)]
