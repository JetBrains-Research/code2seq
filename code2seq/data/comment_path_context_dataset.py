from typing import Dict, List, Optional

from code2seq.data.vocabulary import CommentVocabulary
from omegaconf import DictConfig

from code2seq.data.path_context_dataset import PathContextDataset


class CommentPathContextDataset(PathContextDataset):
    def __init__(self, data_file: str, config: DictConfig, vocabulary: CommentVocabulary, random_context: bool):
        super().__init__(data_file, config, vocabulary, random_context)
        self._vocab: CommentVocabulary = vocabulary

    def tokenize_label(self, raw_label: str, vocab: Dict[str, int], max_parts: Optional[int]) -> List[int]:
        tokenizer = self._vocab.tokenizer
        tokenized_snippet = tokenizer(
            raw_label.replace(PathContextDataset._separator, " "),
            add_special_tokens=True,
            padding="max_length" if max_parts else "do_not_pad",
            max_length=max_parts,
        )
        return tokenized_snippet["input_ids"]
