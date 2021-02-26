from dataclasses import dataclass
from typing import Dict, List

import numpy
import torch

from omegaconf import DictConfig

# path context keys
FROM_TYPE = "from_type"
FROM_TOKEN = "from_token"
PATH_NODES = "path_nodes"
TO_TOKEN = "to_token"
TO_TYPE = "to_type"


@dataclass
class ContextPart:
    name: str
    to_id: Dict[str, int]
    parameters: DictConfig


@dataclass
class PathContextSample:
    contexts: Dict[str, numpy.ndarray]
    label: numpy.ndarray
    n_contexts: int


class PathContextBatch:
    def __init__(self, samples: List[PathContextSample]):
        samples = [s for s in samples if s is not None]
        self.contexts_per_label = [_s.n_contexts for _s in samples]

        torch_labels = numpy.hstack([_s.label for _s in samples])
        self.labels = torch.from_numpy(torch_labels)

        self.contexts = {}
        for key in samples[0].contexts:
            key_union = numpy.hstack([_s.contexts[key] for _s in samples])
            self.contexts[key] = torch.from_numpy(key_union)

    def __len__(self) -> int:
        return len(self.contexts_per_label)

    def pin_memory(self) -> "PathContextBatch":
        self.labels = self.labels.pin_memory()
        for key, tensor in self.contexts.items():
            self.contexts[key] = self.contexts[key].pin_memory()
        return self

    def move_to_device(self, device: torch.device):
        self.labels = self.labels.to(device)
        for key, tensor in self.contexts.items():
            self.contexts[key] = self.contexts[key].to(device)
