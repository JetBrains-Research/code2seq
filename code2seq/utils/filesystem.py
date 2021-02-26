import subprocess
from os import mkdir, getcwd
from os.path import exists, split, join
from shutil import rmtree
from typing import Generator, List, Tuple


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


def get_test_resources_dir() -> str:
    cur_working_directory = getcwd()
    if split(cur_working_directory)[-1] != "tests":
        cur_working_directory = join(cur_working_directory, "tests")
    return join(cur_working_directory, "resources")
