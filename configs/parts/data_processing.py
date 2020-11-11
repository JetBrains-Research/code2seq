from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ContextDescription:
    """Config for processing context (token in leaves or path).
    Max parts don't include <SOS> and <EOS> tokens.
    None stands for unlimited size.
    """

    max_parts: int
    is_wrapped: bool
    is_splitted: bool
    vocab_size: Optional[int] = None


@dataclass(frozen=True)
class PathContextConfig:
    token_description: ContextDescription
    path_description: ContextDescription
    target_description: ContextDescription


@dataclass(frozen=True)
class TypedPathContextConfig(PathContextConfig):
    type_description: ContextDescription
