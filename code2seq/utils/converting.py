from typing import Dict, List

import numpy

from code2seq.utils.vocabulary import PAD, UNK, SOS, EOS


def parse_token(token: str, is_split: bool, separator: str = "|") -> List[str]:
    return token.split(separator) if is_split else [token]


def strings_to_wrapped_numpy(
    values: List[str],
    to_id: Dict[str, int],
    is_split: bool,
    max_length: int,
    is_wrapped: bool = False,
    separator: str = "|",
) -> numpy.ndarray:
    """Convert list of string to numpy array with vocabulary ids.
    Also wrap each string with SOS and EOS tokens if needed.

    :param values: list of string
    :param to_id: vocabulary from string to int
    :param is_split: bool flag for splitting token by separator
    :param max_length: max length of subtoken sequence (result array may be max_length + 1 if wrapping is needed)
    :param is_wrapped: bool flag for wrapping each token
    :param separator: string value for separating token into subtokens (my|super|name -> [my, super, name])
    :return: numpy array of shape [max length; number of samples]
    """
    pad_token = to_id[PAD]
    unk_token = to_id[UNK]
    sos_token = to_id.get(SOS, None)
    eos_token = to_id.get(EOS, None)
    if is_wrapped and (sos_token is None or eos_token is None):
        raise ValueError(f"Pass SOS and EOS tokens for wrapping list of tokens")

    size = max_length + (1 if is_wrapped else 0)
    wrapped_numpy = numpy.full((size, len(values)), pad_token, dtype=numpy.int64)

    start_index = 0
    if is_wrapped:
        wrapped_numpy[0, :] = sos_token
        start_index = 1

    for i, value in enumerate(values):
        tokens = parse_token(value, is_split, separator)
        length = min(len(tokens), max_length)
        wrapped_numpy[start_index : start_index + length, i] = [to_id.get(_t, unk_token) for _t in tokens[:length]]
        if is_wrapped and length < max_length:
            wrapped_numpy[start_index + length, i] = eos_token
    return wrapped_numpy
