from dataclasses import dataclass
from typing import List, Iterable, Tuple, Optional

import torch


@dataclass
class Path:
    from_token: torch.Tensor  # [max token parts]
    path_node: torch.Tensor  # [path length]
    to_token: torch.Tensor  # [max token parts]


@dataclass
class LabeledPathContext:
    label: torch.Tensor  # [max label parts]
    path_contexts: List[Path]


class BatchedLabeledPathContext:
    def __init__(self, samples: List[Optional[LabeledPathContext]]):
        samples = [s for s in samples if s is not None]

        # [batch size; max label parts]
        self.labels = torch.cat([s.label for s in samples], dim=1)
        # [batch size]
        self.contexts_per_label = [len(s.path_contexts) for s in samples]

        # [paths in batch; max token parts]
        self.from_token = torch.cat([path.from_token for s in samples for path in s.path_contexts], dim=1)
        # [paths in batch; path length]
        self.path_node = torch.cat([path.path_node for s in samples for path in s.path_contexts], dim=1)
        # [paths in batch; max token parts]
        self.to_token = torch.cat([path.to_token for s in samples for path in s.path_contexts], dim=1)

    def __len__(self) -> int:
        return len(self.contexts_per_label)

    def __get_all_tensors(self) -> Iterable[Tuple[str, torch.Tensor]]:
        for name, value in vars(self).items():
            if isinstance(value, torch.Tensor):
                yield name, value

    def pin_memory(self) -> "BatchedLabeledPathContext":
        for name, value in self.__get_all_tensors():
            setattr(self, name, value.pin_memory())
        return self

    def move_to_device(self, device: torch.device):
        for name, value in self.__get_all_tensors():
            setattr(self, name, value.to(device))


@dataclass
class TypedPath(Path):
    from_type: torch.Tensor  # [max type parts]
    to_type: torch.Tensor  # [max type parts]


@dataclass
class LabeledTypedPathContext(LabeledPathContext):
    path_contexts: List[TypedPath]


class BatchedLabeledTypedPathContext(BatchedLabeledPathContext):
    def __init__(self, samples: List[Optional[LabeledTypedPathContext]]):
        super().__init__(samples)
        # [paths in batch; max type parts]
        self.from_type = torch.cat([path.from_type for s in samples for path in s.path_contexts], dim=1)
        # [paths in batch; max type parts]
        self.to_type = torch.cat([path.to_type for s in samples for path in s.path_contexts], dim=1)
