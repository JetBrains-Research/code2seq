from dataset import Vocabulary, ConvertParameters, BufferedPathContext
from configs.preprocessing_config import PreprocessingConfig

from typing import List, Generator, Tuple, Callable, Any
from utils.common import FROM_TOKEN, PATH_TYPES, TO_TOKEN
from math import ceil
from functools import partial
from multiprocessing import Pool
from utils.common import count_lines_in_file, create_folder
from tqdm import tqdm
from os.path import join, exists

DATA_FOLDER = "data"


# Buffering utils
def read_file_by_batch(filepath: str, batch_size: int) -> Generator[List[str], None, None]:
    with open(filepath, "r") as file:
        lines = []
        for line in file:
            lines.append(line.strip())
            if len(lines) == batch_size:
                yield lines
                lines = []
    yield lines

def split_context(
    line: str, vocab: Vocabulary, convert_path_context_to_ids: Any, **kwargs
) -> Tuple[List[int], List[Tuple[List[int], List[int], List[int]]]]:
    label, *path_contexts = line.split()
    converted_context = [convert_path_context_to_ids(pc, vocab, **kwargs) for pc in path_contexts]
    return [vocab.label_to_id[label]], converted_context


def convert_raw_buffer(
    convert_args: Tuple[List[str], PreprocessingConfig, Vocabulary, str, Any,],
    **kwargs,
):
    lines, config, vocab, output_path, convert_path_context_to_ids = convert_args
    labels, from_tokens, path_types, to_tokens = [], [], [], []
    for line in lines:
        label_id, converted_context = split_context(line, vocab, convert_path_context_to_ids, **kwargs)
        labels.append(label_id)
        from_tokens.append([cc[0] for cc in converted_context])
        path_types.append([cc[1] for cc in converted_context])
        to_tokens.append([cc[2] for cc in converted_context])

    BufferedPathContext.create_from_lists(
        (labels, ConvertParameters(config.max_target_parts, config.wrap_target, vocab.label_to_id),),
        {
            FROM_TOKEN: (from_tokens, ConvertParameters(config.max_name_parts, config.wrap_name, vocab.token_to_id),),
            PATH_TYPES: (path_types, ConvertParameters(config.max_path_length, config.wrap_path, vocab.type_to_id),),
            TO_TOKEN: (to_tokens, ConvertParameters(config.max_name_parts, config.wrap_name, vocab.token_to_id),),
        },
    ).dump(output_path)


def convert_holdout(
    holdout_data_path: str,
    holdout_output_folder: str,
    vocab: Vocabulary,
    config: PreprocessingConfig,
    n_jobs: int,
    split_context: Callable[[Any, Any, Any], Any],
    **kwargs,
) -> None:
    if not exists(holdout_output_folder):
        create_folder(holdout_output_folder)

    n_buffers = ceil(count_lines_in_file(holdout_data_path) / config.buffer_size)
    with Pool(n_jobs) as pool:
        results = pool.imap(
            partial(convert_raw_buffer, **kwargs),
            (
                (lines, config, vocab, join(holdout_output_folder, f"buffered_paths_{pos}.pkl"), split_context,)
                for pos, lines in enumerate(read_file_by_batch(holdout_data_path, config.buffer_size))
            ),
        )
        _ = [_ for _ in tqdm(results, total=n_buffers)]
