from dataclasses import dataclass
from typing import Dict, List

import numpy
import torch


@dataclass
class PathContextSample:
    contexts: Dict[str, numpy.ndarray]
    label: numpy.ndarray
    n_contexts: int


class PathContextBatch:
    def __init__(self, samples: List[PathContextSample]):
        self.contexts_per_label = [_s.n_contexts for _s in samples]

        tensor_labels = [torch.tensor(_s.label, dtype=torch.long) for _s in samples]
        self.labels = torch.cat(tensor_labels, dim=-1)

        self.contexts = {}
        for sample in samples:
            for key, value in sample.contexts.items():
                if key not in self.contexts:
                    self.contexts[key] = []
                self.contexts[key].append(torch.tensor(value, dtype=torch.long))
        for key, tensor_list in self.contexts.items():
            self.contexts[key] = torch.cat(tensor_list, dim=-1)

    def __len__(self) -> int:
        return self.labels.shape[1]

    def pin_memory(self) -> "PathContextBatch":
        self.labels = self.labels.pin_memory()
        for key, tensor in self.contexts.items():
            self.contexts[key] = self.contexts[key].pin_memory()
        return self

    def move_to_device(self, device: torch.device):
        self.labels = self.labels.to(device)
        for key, tensor in self.contexts.items():
            self.contexts[key] = self.contexts[key].to(device)
