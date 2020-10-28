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


def segment_sizes_to_slices(sizes: List) -> List:
    cum_sums = numpy.cumsum(sizes)
    start_of_segments = numpy.append([0], cum_sums[:-1])
    return [slice(start, end) for start, end in zip(start_of_segments, cum_sums)]
