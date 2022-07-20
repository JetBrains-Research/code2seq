from typing import Dict, List, Optional

from transformers import RobertaTokenizerFast

from code2seq.data.path_context_dataset import PathContextDataset

tokenizer = RobertaTokenizerFast.from_pretrained("microsoft/codebert-base")


class CommentPathContextDataset(PathContextDataset):
    @staticmethod
    def tokenize_label(raw_label: str, vocab: Dict[str, int], max_parts: Optional[int]) -> List[int]:
        label_with_spaces = " ".join(raw_label.split(PathContextDataset._separator))
        label_tokens = tokenizer.tokenize(label_with_spaces)
        if max_parts is None:
            max_parts = len(label_tokens)
        label_tokens = [tokenizer.bos_token] + label_tokens[: max_parts - 2] + [tokenizer.eos_token]
        label_tokens += [tokenizer.pad_token] * (max_parts - len(label_tokens))
        return tokenizer.convert_tokens_to_ids(label_tokens)
