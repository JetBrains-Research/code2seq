import pickle
from collections import Counter
from os.path import join, exists, dirname
from typing import Dict, Counter as TCounter, Type

from commode_utils.vocabulary import BaseVocabulary
from tqdm.auto import tqdm

from commode_utils.filesystem import count_lines_in_file
from omegaconf import DictConfig
from transformers import RobertaTokenizerFast

from code2seq.data.comment_path_context_dataset import CommentPathContextDataset
from code2seq.data.path_context_data_module import PathContextDataModule
from code2seq.data.vocabulary import CommentVocabulary


def _build_from_scratch(config: DictConfig, train_data: str, vocabulary_cls: Type[BaseVocabulary]):
    total_samples = count_lines_in_file(train_data)
    counters: Dict[str, TCounter[str]] = {
        key: Counter() for key in [vocabulary_cls.LABEL, vocabulary_cls.TOKEN, vocabulary_cls.NODE]
    }
    with open(train_data, "r") as f_in:
        for raw_sample in tqdm(f_in, total=total_samples):
            vocabulary_cls.process_raw_sample(raw_sample, counters)

    training_corpus = []
    for string, amount in counters[vocabulary_cls.LABEL].items():
        training_corpus.extend([string] * amount)
    old_tokenizer = RobertaTokenizerFast.from_pretrained(config.base_tokenizer)
    if config.train_new_tokenizer:
        tokenizer = old_tokenizer.train_new_from_iterator(training_corpus, config.max_tokenizer_vocab)
    else:
        tokenizer = old_tokenizer

    for feature, counter in counters.items():
        print(f"Count {len(counter)} {feature}, top-5: {counter.most_common(5)}")

    dataset_dir = dirname(train_data)
    vocabulary_file = join(dataset_dir, vocabulary_cls.vocab_filename)
    with open(vocabulary_file, "wb") as f_out:
        pickle.dump(counters, f_out)
        pickle.dump(tokenizer, f_out)


class CommentPathContextDataModule(PathContextDataModule):
    _vocabulary: CommentVocabulary

    def __init__(self, data_dir: str, config: DictConfig):
        super().__init__(data_dir, config)

    def _create_dataset(self, holdout_file: str, random_context: bool) -> CommentPathContextDataset:
        if self._vocabulary is None:
            raise RuntimeError(f"Setup vocabulary before creating data loaders")
        return CommentPathContextDataset(holdout_file, self._config, self._vocabulary, random_context)

    def setup_vocabulary(self) -> CommentVocabulary:
        if not exists(join(self._data_dir, CommentVocabulary.vocab_filename)):
            print("Can't find vocabulary, collect it from train holdout")
            _build_from_scratch(self._config, join(self._data_dir, f"{self._train}.c2s"), CommentVocabulary)
        vocabulary_path = join(self._data_dir, CommentVocabulary.vocab_filename)
        return CommentVocabulary(vocabulary_path, self._config.labels_count, self._config.tokens_count)

    @property
    def vocabulary(self) -> CommentVocabulary:
        if self._vocabulary is None:
            raise RuntimeError(f"Setup data module for initializing vocabulary")
        return self._vocabulary
