from argparse import ArgumentParser
from collections import Counter
from os.path import dirname, join
from pickle import load, dump
from typing import Dict, Counter as CounterType, Optional, List

from commode_utils.vocabulary import BaseVocabulary, build_from_scratch


class Vocabulary(BaseVocabulary):
    def __init__(
        self,
        vocabulary_file: str,
        max_labels: Optional[int] = None,
        max_tokens: Optional[int] = None,
        is_class: bool = False,
    ):
        super().__init__(vocabulary_file, max_labels, max_tokens)
        if is_class:
            self._label_to_id = {
                token[0]: i for i, token in enumerate(self._counters[self.LABEL].most_common(max_labels))
            }

    @staticmethod
    def _process_raw_sample(raw_sample: str, counters: Dict[str, CounterType[str]], context_seq: List[str]):
        label, *path_contexts = raw_sample.split(" ")
        counters[Vocabulary.LABEL].update(label.split(Vocabulary._separator))
        for path_context in path_contexts:
            for token, desc in zip(path_context.split(","), context_seq):
                counters[desc].update(token.split(Vocabulary._separator))

    @staticmethod
    def process_raw_sample(raw_sample: str, counters: Dict[str, CounterType[str]]):
        Vocabulary._process_raw_sample(
            raw_sample, counters, [BaseVocabulary.TOKEN, BaseVocabulary.NODE, BaseVocabulary.TOKEN]
        )


class TypedVocabulary(Vocabulary):
    TYPE = "tokenType"

    _path_context_seq = [TYPE, Vocabulary.TOKEN, Vocabulary.NODE, Vocabulary.TOKEN, TYPE]

    def __init__(
        self,
        vocabulary_file: str,
        max_labels: Optional[int] = None,
        max_tokens: Optional[int] = None,
        max_types: Optional[int] = None,
    ):
        super().__init__(vocabulary_file, max_labels, max_tokens)

        self._type_to_id = {self.PAD: 0, self.UNK: 1, self.SOS: 2, self.EOS: 3}
        self._type_to_id.update(
            (token[0], i + 4) for i, token in enumerate(self._counters[self.TYPE].most_common(max_types))
        )

    @property
    def type_to_id(self) -> Dict[str, int]:
        return self._type_to_id

    @staticmethod
    def process_raw_sample(raw_sample: str, counters: Dict[str, CounterType[str]]):
        if TypedVocabulary.TYPE not in counters:
            counters[TypedVocabulary.TYPE] = Counter()
        context_seq = [
            TypedVocabulary.TYPE,
            BaseVocabulary.TOKEN,
            BaseVocabulary.NODE,
            BaseVocabulary.TOKEN,
            TypedVocabulary.TYPE,
        ]
        TypedVocabulary._process_raw_sample(raw_sample, counters, context_seq)


def convert_from_vanilla(vocabulary_path: str):
    counters: Dict[str, CounterType[str]] = {}
    with open(vocabulary_path, "rb") as dict_file:
        counters[Vocabulary.TOKEN] = Counter(load(dict_file))
        counters[Vocabulary.NODE] = Counter(load(dict_file))
        counters[Vocabulary.LABEL] = Counter(load(dict_file))

    for feature, counter in counters.items():
        print(f"Count {len(counter)} {feature}, top-5: {counter.most_common(5)}")

    dataset_dir = dirname(vocabulary_path)
    vocabulary_file = join(dataset_dir, Vocabulary.vocab_filename)
    with open(vocabulary_file, "wb") as f_out:
        dump(counters, f_out)


if __name__ == "__main__":
    __arg_parse = ArgumentParser()
    __arg_parse.add_argument("data", type=str, help="Path to file with data")
    __arg_parse.add_argument("--typed", action="store_true", help="Use typed vocabulary")
    __args = __arg_parse.parse_args()

    if __args.data.endswith(".dict.c2s"):
        convert_from_vanilla(__args.data)
    else:
        __vocab_cls = TypedVocabulary if __args.typed else Vocabulary
        build_from_scratch(__args.data, __vocab_cls)
