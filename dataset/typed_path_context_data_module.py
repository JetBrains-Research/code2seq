from typing import Optional

from configs.parts import DataProcessingConfig, ModelHyperParameters
from dataset import PathContextDataModule, TypedPathContextDataset
from utils.vocabulary import Vocabulary


class TypedPathContextDataModule(PathContextDataModule):

    _train_dataset: TypedPathContextDataset
    _val_dataset: TypedPathContextDataset
    _test_dataset: TypedPathContextDataset

    def __init__(
        self,
        dataset_name: str,
        vocabulary: Vocabulary,
        data_params: DataProcessingConfig,
        model_params: ModelHyperParameters,
        num_workers: int = 0,
    ):
        super().__init__(dataset_name, vocabulary, data_params, model_params, num_workers)

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self._train_dataset = TypedPathContextDataset(
                self._train_data_file,
                self._vocabulary,
                self._data_config,
                self._hyper_params.max_context,
                self._hyper_params.random_context,
            )
            self._val_dataset = TypedPathContextDataset(
                self._val_data_file, self._vocabulary, self._data_config, self._hyper_params.max_context, False
            )
        else:
            self._test_dataset = TypedPathContextDataset(
                self._test_data_file, self._vocabulary, self._data_config, self._hyper_params.max_context, False
            )
