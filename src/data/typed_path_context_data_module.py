from typing import List, Optional

from omegaconf import DictConfig

from src.data import (
    PathContextDataModule,
    TypedPathContextDataset,
    BatchedLabeledTypedPathContext,
    LabeledTypedPathContext,
)
from src.data.vocabulary import TypedVocabulary


class TypedPathContextDataModule(PathContextDataModule):
    _vocabulary: Optional[TypedVocabulary] = None

    def __init__(self, data_dir: str, config: DictConfig):
        super().__init__(data_dir, config)

    @staticmethod
    def collate_wrapper(batch: List[Optional[LabeledTypedPathContext]]) -> BatchedLabeledTypedPathContext:
        return BatchedLabeledTypedPathContext(batch)

    def _create_dataset(self, holdout_file: str, random_context: bool) -> TypedPathContextDataset:
        return TypedPathContextDataset(holdout_file, self._config, self._vocabulary, random_context)

    @property
    def vocabulary(self) -> TypedVocabulary:
        if self._vocabulary is None:
            raise RuntimeError(f"Setup data module for initializing vocabulary")
        return self._vocabulary
