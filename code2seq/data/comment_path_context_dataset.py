from typing import Dict, List, Optional

from code2seq.data.path_context_dataset import PathContextDataset
from code2seq.data.vocabulary import Vocabulary


class CommentPathContextDataset(PathContextDataset):

    @staticmethod
    def tokenize_label(raw_label: str, vocab: Dict[str, int], max_parts: Optional[int]) -> List[int]:
        sublabels = raw_label.split(PathContextDataset._separator)
        max_parts = max_parts or len(sublabels)
        label_unk = vocab[Vocabulary.UNK]

        label = [vocab[Vocabulary.SOS]] + [vocab.get(st, label_unk) for st in sublabels[:max_parts]]
        if len(sublabels) < max_parts:
            label.append(vocab[Vocabulary.EOS])
            label += [vocab[Vocabulary.PAD]] * (max_parts + 1 - len(label))
        return label
