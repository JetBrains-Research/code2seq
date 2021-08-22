from dataclasses import dataclass
from typing import Iterable, Tuple, Optional, Sequence

import torch


@dataclass
class Path:
    from_token: torch.Tensor  # [max token parts]
    path_node: torch.Tensor  # [path length]
    to_token: torch.Tensor  # [max token parts]


@dataclass
class LabeledPathContext:
    label: torch.Tensor  # [max label parts]
    path_contexts: Sequence[Path]


class BatchedLabeledPathContext:
    def __init__(self, all_samples: Sequence[Optional[LabeledPathContext]]):
        samples = [s for s in all_samples if s is not None]

        # [max label parts; batch size]
        self.labels = torch.cat([s.label.unsqueeze(1) for s in samples], dim=1)
        # [batch size]
        self.contexts_per_label = torch.tensor([len(s.path_contexts) for s in samples])

        # [max token parts; n contexts]
        self.from_token = torch.cat([path.from_token.unsqueeze(1) for s in samples for path in s.path_contexts], dim=1)
        # [path length; n contexts]
        self.path_nodes = torch.cat([path.path_node.unsqueeze(1) for s in samples for path in s.path_contexts], dim=1)
        # [max token parts; n contexts]
        self.to_token = torch.cat([path.to_token.unsqueeze(1) for s in samples for path in s.path_contexts], dim=1)

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
    path_contexts: Sequence[TypedPath]


class BatchedLabeledTypedPathContext(BatchedLabeledPathContext):
    def __init__(self, all_samples: Sequence[Optional[LabeledTypedPathContext]]):
        super().__init__(all_samples)
        samples = [s for s in all_samples if s is not None]
        # [max type parts; n contexts]
        self.from_type = torch.cat([path.from_type.unsqueeze(1) for s in samples for path in s.path_contexts], dim=1)
        # [max type parts; n contexts]
        self.to_type = torch.cat([path.to_type.unsqueeze(1) for s in samples for path in s.path_contexts], dim=1)
