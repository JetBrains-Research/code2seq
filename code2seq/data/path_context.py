from dataclasses import dataclass
from typing import Iterable, Tuple, Optional, Sequence, List, cast

import torch


@dataclass
class Path:
    from_token: List[int]  # [max token parts]
    path_node: List[int]  # [path length]
    to_token: List[int]  # [max token parts]


@dataclass
class LabeledPathContext:
    label: List[int]  # [max label parts]
    path_contexts: Sequence[Path]


def transpose(list_of_lists: List[List[int]]) -> List[List[int]]:
    return [cast(List[int], it) for it in zip(*list_of_lists)]


class BatchedLabeledPathContext:
    def __init__(self, all_samples: Sequence[Optional[LabeledPathContext]]):
        samples = [s for s in all_samples if s is not None]

        # [max label parts; batch size]
        self.labels = torch.tensor(transpose([s.label for s in samples]), dtype=torch.long)
        # [batch size]
        self.contexts_per_label = torch.tensor([len(s.path_contexts) for s in samples])

        # [max token parts; n contexts]
        self.from_token = torch.tensor(
            transpose([path.from_token for s in samples for path in s.path_contexts]), dtype=torch.long
        )
        # [path length; n contexts]
        self.path_nodes = torch.tensor(
            transpose([path.path_node for s in samples for path in s.path_contexts]), dtype=torch.long
        )
        # [max token parts; n contexts]
        self.to_token = torch.tensor(
            transpose([path.to_token for s in samples for path in s.path_contexts]), dtype=torch.long
        )

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
    from_type: List[int]  # [max type parts]
    to_type: List[int]  # [max type parts]


@dataclass
class LabeledTypedPathContext(LabeledPathContext):
    path_contexts: Sequence[TypedPath]


class BatchedLabeledTypedPathContext(BatchedLabeledPathContext):
    def __init__(self, all_samples: Sequence[Optional[LabeledTypedPathContext]]):
        super().__init__(all_samples)
        samples = [s for s in all_samples if s is not None]
        # [max type parts; n contexts]
        self.from_type = torch.tensor(
            transpose([path.from_type for s in samples for path in s.path_contexts]), dtype=torch.long
        )
        # [max type parts; n contexts]
        self.to_type = torch.tensor(
            transpose([path.to_type for s in samples for path in s.path_contexts]), dtype=torch.long
        )
