from typing import Dict, List, Optional

from omegaconf import DictConfig
from transformers import RobertaTokenizerFast

from code2seq.data.path_context_dataset import PathContextDataset
from code2seq.data.vocabulary import Vocabulary


class CommentPathContextDataset(PathContextDataset):
    def __init__(self, data_file: str, config: DictConfig, vocabulary: Vocabulary, random_context: bool):
        super().__init__(data_file, config, vocabulary, random_context)

    @staticmethod
    def tokenize_label(raw_label: str, vocab: Dict[str, int], max_parts: Optional[int]) -> List[int]:
        tokenizer = RobertaTokenizerFast.from_pretrained("microsoft/codebert-base")

        label_with_spaces = ' '.join(raw_label.split(PathContextDataset._separator))
        label_tokens = [tokenizer.bos_token] + tokenizer.tokenize(label_with_spaces)[:max_parts - 2] + [
            tokenizer.eos_token]
        label_tokens += [tokenizer.pad_token] * (max_parts - len(label_tokens))

        return tokenizer.convert_tokens_to_ids(label_tokens)
