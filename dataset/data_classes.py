from dataclasses import dataclass
from typing import Dict

import numpy


@dataclass
class PathContextSample:
    contexts: Dict[str, numpy.ndarray]
    label: numpy.ndarray
    n_contexts: int
