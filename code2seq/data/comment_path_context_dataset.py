from typing import Dict, List, Optional

from code2seq.data.vocabulary import CommentVocabulary
from omegaconf import DictConfig

from code2seq.data.path_context_dataset import PathContextDataset


class CommentPathContextDataset(PathContextDataset):
    def __init__(self, data_file: str, config: DictConfig, vocabulary: CommentVocabulary, random_context: bool):
        super().__init__(data_file, config, vocabulary, random_context)
        self._vocab: CommentVocabulary = vocabulary

    def tokenize_label(self, raw_label: str, vocab: Dict[str, int], max_parts: Optional[int]) -> List[int]:
        label_with_spaces = " ".join(raw_label.split(PathContextDataset._separator))
        tokenizer = self._vocab.tokenizer
        label_tokens = tokenizer.tokenize(label_with_spaces)
        if max_parts is None:
            max_parts = len(label_tokens)
        label_tokens = [tokenizer.bos_token] + label_tokens[: max_parts - 2] + [tokenizer.eos_token]
        label_tokens += [tokenizer.pad_token] * (max_parts - len(label_tokens))
        print(label_tokens)
        return tokenizer.convert_tokens_to_ids(label_tokens)
