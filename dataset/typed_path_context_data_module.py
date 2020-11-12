from omegaconf import DictConfig
from torch.utils.data import Dataset

from dataset import PathContextDataModule, TypedPathContextDataset
from utils.vocabulary import Vocabulary


class TypedPathContextDataModule(PathContextDataModule):
    def __init__(self, config: DictConfig, vocabulary: Vocabulary):
        super().__init__(config, vocabulary)

    def _create_dataset(self, data_file: str, random_context: bool) -> Dataset:
        return TypedPathContextDataset(data_file, self._config, self._vocabulary, random_context)
