from typing import Dict, List

import numpy

from utils.common import UNK, PAD, SOS, EOS


def parse_token(token: str, is_split: bool, separator: str = None) -> List[str]:
    return token.split(separator) if is_split else [token]


def str_to_list(value: str, to_id: Dict[str, int], is_split: bool, separator: str) -> List[int]:
    value_list = parse_token(value, is_split, separator)
    unk_id = to_id[UNK]
    return [to_id.get(_v, unk_id) for _v in value_list]


def list_to_wrapped_numpy(
    values: List[int], to_id: Dict[str, int], max_length: int, is_wrapped: bool = False,
) -> numpy.ndarray:
    pad_token = to_id[PAD]
    sos_token = to_id.get(SOS, None)
    eos_token = to_id.get(EOS, None)
    if is_wrapped and (sos_token is None or eos_token is None):
        raise ValueError(f"Pass SOS and EOS tokens for wrapping list of tokens")
    size = max_length + (1 if is_wrapped else 0)
    wrapped_numpy = numpy.full((size, 1), pad_token, dtype=numpy.int32)
    start_index = 0
    if is_wrapped:
        wrapped_numpy[0] = sos_token
        start_index += 1
    length = min(len(values), max_length)
    wrapped_numpy[start_index : start_index + length, 0] = values[:length]
    if is_wrapped and length < max_length:
        wrapped_numpy[start_index + length] = eos_token
    return wrapped_numpy
