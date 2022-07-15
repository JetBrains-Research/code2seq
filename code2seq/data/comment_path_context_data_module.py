from code2seq.data.comment_path_context_dataset import CommentPathContextDataset
from code2seq.data.path_context_data_module import PathContextDataModule


class CommentPathContextDataModule(PathContextDataModule):
    def _create_dataset(self, holdout_file: str, random_context: bool) -> CommentPathContextDataset:
        if self._vocabulary is None:
            raise RuntimeError(f"Setup vocabulary before creating data loaders")
        return CommentPathContextDataset(holdout_file, self._config, self._vocabulary, random_context)
